import torch
import torch.nn.functional as F
from torch import nn

from src.model.utils.box3d_utils import circle_nms, xywhr2xyxyr
from src.model.utils.lidar_box3d import LiDARInstance3DBoxes
from src.ops import nms_gpu

from .utils import (
    boxes_to_local,
    decode_box_deltas,
    encode_box_deltas,
    local_to_global,
    normalize_bev_points,
    pairwise_bev_iou,
    points_in_boxes_mask,
    project_points_to_image,
)


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x):
        return self.net(x)


class AttentionPooling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, tokens, mask):
        if tokens.numel() == 0:
            return tokens.new_zeros((tokens.size(0), tokens.size(-1)))
        logits = self.score(tokens).squeeze(-1)
        logits = logits.masked_fill(~mask, float('-inf'))
        valid_rows = mask.any(dim=1)
        logits = torch.where(valid_rows.unsqueeze(-1), logits, logits.new_zeros(logits.shape))
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(valid_rows.unsqueeze(-1), weights, weights.new_zeros(weights.shape))
        return torch.bmm(weights.unsqueeze(1), tokens).squeeze(1)


class CrossModalDeformableAttentionLite(nn.Module):
    def __init__(self, query_dim, feat_dim, hidden_dim, num_samples=4, offset_scale=0.05):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_samples = int(num_samples)
        self.offset_scale = float(offset_scale)
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.feat_proj = nn.Conv2d(feat_dim, hidden_dim, kernel_size=1)
        self.offset_proj = nn.Linear(query_dim, self.num_samples * 2)
        self.weight_proj = nn.Linear(query_dim, self.num_samples)
        self.out_proj = MLP(hidden_dim * 2, hidden_dim, hidden_dim)

    def forward(self, queries, ref_points, feature_map, valid_mask=None):
        if queries.numel() == 0:
            return queries.new_zeros((queries.size(0), queries.size(1), self.hidden_dim))
        projected_queries = self.query_proj(queries)
        projected_feats = self.feat_proj(feature_map)
        offsets = torch.tanh(self.offset_proj(queries)).view(queries.size(0), queries.size(1), self.num_samples, 2)
        offsets = offsets * self.offset_scale
        weights = torch.softmax(self.weight_proj(queries), dim=-1)
        sample_grid = (ref_points.unsqueeze(2) + offsets).clamp(-1.2, 1.2)
        sampled = F.grid_sample(
            projected_feats,
            sample_grid.reshape(queries.size(0), -1, 1, 2),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )
        sampled = sampled.reshape(queries.size(0), self.hidden_dim, queries.size(1), self.num_samples)
        sampled = sampled.permute(0, 2, 3, 1)
        if valid_mask is not None:
            sampled = sampled * valid_mask[:, :, None, None].to(sampled.dtype)
            weights = weights * valid_mask[:, :, None].to(weights.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        context = (sampled * weights.unsqueeze(-1)).sum(dim=2)
        if valid_mask is not None:
            context = context * valid_mask.unsqueeze(-1).to(context.dtype)
            projected_queries = projected_queries * valid_mask.unsqueeze(-1).to(projected_queries.dtype)
        return self.out_proj(torch.cat([projected_queries, context], dim=-1))


class ProposalPointGuidedFusion(nn.Module):
    def __init__(
        self,
        point_in_channels=7,
        image_feat_dim=96,
        hidden_dim=128,
        max_points=32,
        num_samples=4,
        offset_scale=0.04,
    ):
        super().__init__()
        self.point_in_channels = point_in_channels
        self.hidden_dim = hidden_dim
        self.max_points = int(max_points)
        self.point_encoder = MLP(point_in_channels + 1, hidden_dim, hidden_dim)
        self.cmda = CrossModalDeformableAttentionLite(
            query_dim=hidden_dim,
            feat_dim=image_feat_dim,
            hidden_dim=hidden_dim,
            num_samples=num_samples,
            offset_scale=offset_scale,
        )
        self.pool = AttentionPooling(hidden_dim)

    def _estimate_density(self, local_points):
        if local_points.size(0) <= 1:
            return local_points.new_ones((local_points.size(0), 1))
        distance = torch.cdist(local_points, local_points)
        knn = distance.topk(k=min(4, local_points.size(0)), largest=False).values
        mean_distance = knn[:, 1:].mean(dim=1, keepdim=True) if knn.size(1) > 1 else knn[:, :1]
        return torch.exp(-mean_distance)

    def forward(self, points, boxes, image_feat, meta):
        if boxes.numel() == 0:
            return points.new_zeros((0, self.hidden_dim)), points.new_zeros((0, 1))

        point_masks = points_in_boxes_mask(points[:, :3], boxes)
        proposal_features = []
        point_counts = []
        for proposal_idx in range(boxes.size(0)):
            inside_mask = point_masks[proposal_idx]
            proposal_points = points[inside_mask]
            point_counts.append(points.new_tensor([[proposal_points.size(0)]], dtype=torch.float32))
            if proposal_points.numel() == 0:
                proposal_features.append(points.new_zeros(self.hidden_dim))
                continue

            local_xyz = boxes_to_local(proposal_points[:, :3], boxes[proposal_idx:proposal_idx + 1]).squeeze(0)
            normalized_local = local_xyz.clone()
            normalized_local[:, 0] /= boxes[proposal_idx, 3].clamp(min=1e-3)
            normalized_local[:, 1] /= boxes[proposal_idx, 4].clamp(min=1e-3)
            normalized_local[:, 2] = normalized_local[:, 2] / boxes[proposal_idx, 5].clamp(min=1e-3) - 0.5
            distance = normalized_local[:, :2].norm(dim=-1)
            topk = min(self.max_points, proposal_points.size(0))
            keep = torch.topk(distance, k=topk, largest=False).indices
            proposal_points = proposal_points[keep]
            normalized_local = normalized_local[keep]
            density = self._estimate_density(normalized_local)

            point_tokens = torch.cat([normalized_local, proposal_points[:, 3:self.point_in_channels], density], dim=-1)
            point_tokens = self.point_encoder(point_tokens).unsqueeze(0)
            ref_points, valid = project_points_to_image(proposal_points[:, :3], meta)
            image_tokens = self.cmda(
                point_tokens,
                ref_points.unsqueeze(0),
                image_feat,
                valid.unsqueeze(0),
            )
            fused_tokens = point_tokens + image_tokens
            pooled = self.pool(
                fused_tokens,
                torch.ones((1, fused_tokens.size(1)), device=fused_tokens.device, dtype=torch.bool),
            ).squeeze(0)
            proposal_features.append(pooled)

        return torch.stack(proposal_features, dim=0), torch.cat(point_counts, dim=0)


class ProposalGridGuidedFusion(nn.Module):
    def __init__(
        self,
        image_feat_dim=96,
        bev_feat_dim=384,
        hidden_dim=128,
        grid_size=3,
        num_samples_fv=4,
        num_samples_bev=4,
        fv_offset_scale=0.04,
        bev_offset_scale=0.08,
        point_cloud_range=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_size = int(grid_size)
        self.point_cloud_range = list(point_cloud_range)
        self.register_buffer('local_grid', self._create_local_grid(), persistent=False)
        self.grid_encoder = MLP(3, hidden_dim, hidden_dim)
        self.fv_cmda = CrossModalDeformableAttentionLite(
            query_dim=hidden_dim,
            feat_dim=image_feat_dim,
            hidden_dim=hidden_dim,
            num_samples=num_samples_fv,
            offset_scale=fv_offset_scale,
        )
        self.bev_cmda = CrossModalDeformableAttentionLite(
            query_dim=hidden_dim,
            feat_dim=bev_feat_dim,
            hidden_dim=hidden_dim,
            num_samples=num_samples_bev,
            offset_scale=bev_offset_scale,
        )
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = MLP(hidden_dim, hidden_dim * 2, hidden_dim)
        self.pool = AttentionPooling(hidden_dim)

    def _create_local_grid(self):
        coords = torch.linspace(-0.5 + 0.5 / self.grid_size, 0.5 - 0.5 / self.grid_size, self.grid_size)
        zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing='ij')
        return torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    def forward(self, boxes, image_feat, bev_feat, meta):
        if boxes.numel() == 0:
            return boxes.new_zeros((0, self.hidden_dim))

        local_grid = self.local_grid.to(device=boxes.device, dtype=boxes.dtype)
        normalized_local = local_grid.unsqueeze(0).expand(boxes.size(0), -1, -1).clone()
        grid_offsets = normalized_local.clone()
        grid_offsets[:, :, 0] *= boxes[:, None, 3]
        grid_offsets[:, :, 1] *= boxes[:, None, 4]
        grid_offsets[:, :, 2] = (normalized_local[:, :, 2] + 0.5) * boxes[:, None, 5]
        world_grid = local_to_global(grid_offsets, boxes)

        query_tokens = self.grid_encoder(normalized_local)
        flat_world = world_grid.reshape(-1, 3)
        image_ref, image_valid = project_points_to_image(flat_world, meta)
        image_tokens = self.fv_cmda(
            query_tokens.reshape(1, -1, self.hidden_dim),
            image_ref.reshape(1, -1, 2),
            image_feat,
            image_valid.reshape(1, -1),
        ).reshape(boxes.size(0), -1, self.hidden_dim)

        bev_ref, bev_valid = normalize_bev_points(flat_world[:, :2], self.point_cloud_range)
        bev_tokens = self.bev_cmda(
            image_tokens.reshape(1, -1, self.hidden_dim),
            bev_ref.reshape(1, -1, 2),
            bev_feat,
            bev_valid.reshape(1, -1),
        ).reshape(boxes.size(0), -1, self.hidden_dim)

        tokens = query_tokens + image_tokens + bev_tokens
        attn_tokens, _ = self.self_attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_tokens)
        tokens = self.norm2(tokens + self.ffn(tokens))
        mask = torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
        return self.pool(tokens, mask)


class ProposalRefinementHead(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=3):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 16)
        self.geometry_encoder = MLP(9, hidden_dim, hidden_dim)
        self.fusion = MLP(hidden_dim * 3 + 16 + 1, hidden_dim * 2, hidden_dim)
        self.quality = nn.Linear(hidden_dim, 1)
        self.box_delta = nn.Linear(hidden_dim, 7)

    def forward(self, pgf_feat, ggf_feat, proposals, scores, labels, point_counts, point_cloud_range):
        x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
        centers = proposals[:, :3].clone()
        centers[:, 0] = (centers[:, 0] - x_min) / max(x_max - x_min, 1e-5)
        centers[:, 1] = (centers[:, 1] - y_min) / max(y_max - y_min, 1e-5)
        centers[:, 2] = (centers[:, 2] - z_min) / max(z_max - z_min, 1e-5)
        dims = torch.log(proposals[:, 3:6].clamp(min=1e-3))
        yaw = torch.stack([torch.sin(proposals[:, 6]), torch.cos(proposals[:, 6])], dim=-1)
        geometry = torch.cat([centers, dims, yaw, point_counts.clamp(max=32.0) / 32.0], dim=-1)
        geometry_feat = self.geometry_encoder(geometry)
        label_feat = self.label_embed(labels.clamp(min=0))
        fused = torch.cat([pgf_feat, ggf_feat, geometry_feat, label_feat, scores.unsqueeze(-1)], dim=-1)
        fused = self.fusion(fused)
        return self.quality(fused), self.box_delta(fused)


class CVFusionProposalRefiner(nn.Module):
    def __init__(
        self,
        point_cloud_range,
        image_feat_dim=96,
        bev_feat_dim=384,
        num_classes=3,
        max_train_proposals=96,
        max_test_proposals=64,
        proposal_score_threshold=0.05,
        append_gt_proposals=True,
        gt_center_noise=0.35,
        gt_size_noise=0.1,
        gt_yaw_noise=0.15,
        pos_iou_thr=0.55,
        neg_iou_thr=0.25,
        pgf=None,
        ggf=None,
        refine_head=None,
    ):
        super().__init__()
        self.point_cloud_range = list(point_cloud_range)
        self.max_train_proposals = int(max_train_proposals)
        self.max_test_proposals = int(max_test_proposals)
        self.proposal_score_threshold = float(proposal_score_threshold)
        self.append_gt_proposals = bool(append_gt_proposals)
        self.gt_center_noise = float(gt_center_noise)
        self.gt_size_noise = float(gt_size_noise)
        self.gt_yaw_noise = float(gt_yaw_noise)
        self.pos_iou_thr = float(pos_iou_thr)
        self.neg_iou_thr = float(neg_iou_thr)

        pgf_cfg = dict(pgf or {})
        ggf_cfg = dict(ggf or {})
        refine_cfg = dict(refine_head or {})
        self.point_guided = ProposalPointGuidedFusion(image_feat_dim=image_feat_dim, **pgf_cfg)
        self.grid_guided = ProposalGridGuidedFusion(
            image_feat_dim=image_feat_dim,
            bev_feat_dim=bev_feat_dim,
            point_cloud_range=self.point_cloud_range,
            **ggf_cfg,
        )
        self.refine_head = ProposalRefinementHead(num_classes=num_classes, **refine_cfg)

    def _merge_image_pyramid(self, image_feats):
        fused = image_feats[0]
        if len(image_feats) == 1:
            return fused
        merged = fused
        for feat in image_feats[1:]:
            merged = merged + F.interpolate(
                feat,
                size=fused.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        return merged / float(len(image_feats))

    def _jitter_boxes(self, boxes):
        if boxes.numel() == 0:
            return boxes
        jittered = boxes.clone()
        jittered[:, :3] += torch.randn_like(jittered[:, :3]) * self.gt_center_noise
        jittered[:, 3:6] *= (1.0 + torch.randn_like(jittered[:, 3:6]) * self.gt_size_noise).clamp(min=0.7, max=1.3)
        jittered[:, 6] += torch.randn_like(jittered[:, 6]) * self.gt_yaw_noise
        return jittered

    def build_proposals(self, stage1_dets, gt_bboxes_3d=None, gt_labels_3d=None, training=False):
        proposals = []
        for batch_idx, (bboxes, scores, labels) in enumerate(stage1_dets):
            if hasattr(bboxes, 'tensor'):
                proposal_boxes = bboxes.tensor.detach()
            else:
                proposal_boxes = bboxes.detach()
            proposal_scores = scores.detach()
            proposal_labels = labels.detach().long()

            keep = proposal_scores >= self.proposal_score_threshold
            proposal_boxes = proposal_boxes[keep]
            proposal_scores = proposal_scores[keep]
            proposal_labels = proposal_labels[keep]
            if proposal_scores.numel() > 0:
                order = torch.argsort(proposal_scores, descending=True)
                limit = self.max_train_proposals if training else self.max_test_proposals
                order = order[:limit]
                proposal_boxes = proposal_boxes[order]
                proposal_scores = proposal_scores[order]
                proposal_labels = proposal_labels[order]

            if training and self.append_gt_proposals and gt_bboxes_3d is not None and gt_labels_3d is not None:
                gt_boxes = gt_bboxes_3d[batch_idx].tensor.to(proposal_boxes.device)
                gt_labels = gt_labels_3d[batch_idx].to(proposal_boxes.device)
                gt_boxes = self._jitter_boxes(gt_boxes)
                gt_scores = torch.ones(gt_boxes.size(0), device=proposal_boxes.device)
                if proposal_boxes.numel() == 0:
                    proposal_boxes = gt_boxes
                    proposal_scores = gt_scores
                    proposal_labels = gt_labels
                else:
                    proposal_boxes = torch.cat([proposal_boxes, gt_boxes], dim=0)
                    proposal_scores = torch.cat([proposal_scores, gt_scores], dim=0)
                    proposal_labels = torch.cat([proposal_labels, gt_labels], dim=0)

            proposals.append(
                dict(
                    boxes=proposal_boxes,
                    scores=proposal_scores,
                    labels=proposal_labels,
                )
            )
        return proposals

    def forward(self, proposals, points, image_feats, bev_feat, metas):
        outputs = []
        image_feat = self._merge_image_pyramid(image_feats)
        for batch_idx, proposal in enumerate(proposals):
            boxes = proposal['boxes']
            if boxes.numel() == 0:
                outputs.append(dict(
                    quality=boxes.new_zeros((0, 1)),
                    delta=boxes.new_zeros((0, 7)),
                ))
                continue

            pgf_feat, point_counts = self.point_guided(
                points[batch_idx].to(bev_feat.device),
                boxes,
                image_feat[batch_idx:batch_idx + 1],
                metas[batch_idx],
            )
            ggf_feat = self.grid_guided(
                boxes,
                image_feat[batch_idx:batch_idx + 1],
                bev_feat[batch_idx:batch_idx + 1],
                metas[batch_idx],
            )
            quality, delta = self.refine_head(
                pgf_feat,
                ggf_feat,
                boxes,
                proposal['scores'],
                proposal['labels'],
                point_counts.to(boxes.device),
                self.point_cloud_range,
            )
            outputs.append(dict(quality=quality, delta=delta))
        return outputs

    def loss(self, outputs, proposals, gt_bboxes_3d):
        cls_losses = []
        bbox_losses = []
        num_pos = 0
        for proposal, output, gt_boxes in zip(proposals, outputs, gt_bboxes_3d):
            boxes = proposal['boxes']
            if boxes.numel() == 0:
                continue
            gt_tensor = gt_boxes.tensor.to(boxes.device)
            if gt_tensor.numel() == 0:
                quality_target = torch.zeros((boxes.size(0), 1), device=boxes.device)
                cls_losses.append(F.binary_cross_entropy_with_logits(output['quality'], quality_target, reduction='mean'))
                bbox_losses.append(boxes.new_zeros(()))
                continue

            ious = pairwise_bev_iou(boxes, gt_tensor)
            matched_iou, matched_gt = ious.max(dim=1)
            positive = matched_iou >= self.pos_iou_thr
            negative = matched_iou < self.neg_iou_thr
            valid = positive | negative

            if bool(valid.any().item()):
                quality_target = torch.where(
                    positive.unsqueeze(-1),
                    matched_iou.unsqueeze(-1),
                    matched_iou.new_zeros((matched_iou.size(0), 1)),
                )
                cls_losses.append(
                    F.binary_cross_entropy_with_logits(
                        output['quality'][valid],
                        quality_target[valid],
                        reduction='mean',
                    )
                )
            else:
                cls_losses.append(boxes.new_zeros(()))

            if bool(positive.any().item()):
                target_boxes = gt_tensor[matched_gt[positive]]
                target_delta = encode_box_deltas(boxes[positive], target_boxes)
                bbox_losses.append(F.smooth_l1_loss(output['delta'][positive], target_delta, reduction='mean'))
                num_pos += int(positive.sum().item())
            else:
                bbox_losses.append(boxes.new_zeros(()))

        if not cls_losses:
            zero = next(self.parameters()).new_zeros(())
            return {
                'stage2.loss_cls': zero,
                'stage2.loss_bbox': zero,
                'stage2.num_pos': zero,
            }
        mean_cls = torch.stack([loss if loss.ndim == 0 else loss.mean() for loss in cls_losses]).mean()
        mean_bbox = torch.stack([loss if loss.ndim == 0 else loss.mean() for loss in bbox_losses]).mean()
        return dict(
            **{
                'stage2.loss_cls': mean_cls,
                'stage2.loss_bbox': mean_bbox,
                'stage2.num_pos': mean_cls.new_tensor(float(num_pos)),
            }
        )

    def predict(self, outputs, proposals):
        predictions = []
        for proposal, output in zip(proposals, outputs):
            boxes = proposal['boxes']
            if boxes.numel() == 0:
                predictions.append(dict(
                    boxes=boxes.new_zeros((0, 7)),
                    scores=boxes.new_zeros((0,)),
                    labels=proposal['labels'].new_zeros((0,)),
                ))
                continue
            quality = torch.sigmoid(output['quality'].squeeze(-1))
            refined_boxes = decode_box_deltas(boxes, output['delta'] * quality.unsqueeze(-1))
            refined_scores = torch.sqrt(proposal['scores'].clamp(min=0.0) * quality.clamp(min=0.0))
            predictions.append(dict(
                boxes=refined_boxes,
                scores=refined_scores,
                labels=proposal['labels'],
            ))
        return predictions

    def postprocess(self, predictions, min_radius, score_threshold=0.05, post_max_size=83, nms_thr=0.01):
        final_results = []
        for pred in predictions:
            boxes = pred['boxes']
            scores = pred['scores']
            labels = pred['labels']
            if boxes.numel() == 0:
                final_results.append([
                    LiDARInstance3DBoxes(boxes, box_dim=7),
                    scores,
                    labels,
                ])
                continue

            score_mask = scores >= score_threshold
            boxes = boxes[score_mask]
            scores = scores[score_mask]
            labels = labels[score_mask]
            keep_indices = []
            for label in labels.unique(sorted=True):
                class_mask = labels == label
                class_boxes = boxes[class_mask]
                class_scores = scores[class_mask]
                if class_boxes.numel() == 0:
                    continue
                class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze(1)
                if class_boxes.device.type == 'cuda':
                    boxes_for_nms = xywhr2xyxyr(LiDARInstance3DBoxes(class_boxes, box_dim=7).bev)
                    keep_local = nms_gpu(
                        boxes_for_nms,
                        class_scores,
                        thresh=nms_thr,
                        post_max_size=post_max_size,
                    )
                    keep_indices.append(class_indices[keep_local])
                    continue

                dets = torch.cat([class_boxes[:, :2], class_scores.unsqueeze(-1)], dim=-1)
                keep_local = torch.tensor(
                    circle_nms(
                        dets.detach().cpu().numpy(),
                        min_radius[int(label.item())],
                        post_max_size=post_max_size,
                    ),
                    dtype=torch.long,
                    device=boxes.device,
                )
                keep_indices.append(class_indices[keep_local])

            if keep_indices:
                keep = torch.cat(keep_indices, dim=0)
                keep = keep[torch.argsort(scores[keep], descending=True)]
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
            else:
                boxes = boxes[:0]
                scores = scores[:0]
                labels = labels[:0]

            final_results.append([
                LiDARInstance3DBoxes(boxes, box_dim=7),
                scores,
                labels,
            ])
        return final_results
