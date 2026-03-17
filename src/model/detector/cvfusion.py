from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist

from vod.frame import FrameDataLoader, FrameTransformMatrix, homogeneous_transformation

from src.model.cvfusion import CVFusionImageBackbone, CrossViewBEVLifter, RGIterBEVFusion, CVFusionProposalRefiner
from src.model.detector.centerpoint import CenterPoint


class CVFusion(CenterPoint):
    def __init__(self, config):
        super().__init__(config)

        if not self.use_camera:
            raise ValueError('CVFusion requires `use_camera: true`.')

        stage1_cfg = dict(config.get('stage1', {}))
        stage2_cfg = dict(config.get('stage2', {}))
        loss_weights = dict(config.get('loss_weights', {}))

        image_cfg = dict(stage1_cfg.get('image_backbone', {}))
        lifter_cfg = dict(stage1_cfg.get('bev_lifter', {}))
        rg_iter_cfg = dict(stage1_cfg.get('rg_iter_fusion', {}))

        self.image_backbone = CVFusionImageBackbone(**image_cfg)
        radar_channels = list(config.get('backbone', {}).get('out_channels', []))
        camera_channels = int(image_cfg.get('out_channels', 96))
        lifter_cfg.setdefault('feat_channels', camera_channels)
        self.bev_lifter = CrossViewBEVLifter(
            point_cloud_range=config.get('point_cloud_range'),
            **lifter_cfg,
        )
        self.rg_iter_fusion = RGIterBEVFusion(
            radar_channels=radar_channels,
            camera_channels=camera_channels,
            **rg_iter_cfg,
        )

        neck_out_channels = config.get('neck', {}).get('out_channels', [])
        bev_feat_dim = neck_out_channels if isinstance(neck_out_channels, int) else sum(neck_out_channels)
        self.final_score_threshold = float(
            stage2_cfg.pop('final_score_threshold', self.head.test_cfg.get('score_threshold', 0.1))
        )
        self.proposal_refiner = CVFusionProposalRefiner(
            point_cloud_range=config.get('point_cloud_range'),
            image_feat_dim=camera_channels,
            bev_feat_dim=bev_feat_dim,
            num_classes=len(config.get('class_names', [])),
            **stage2_cfg,
        )

        self.stage1_loss_weight = float(loss_weights.get('stage1', 1.0))
        self.stage2_cls_weight = float(loss_weights.get('stage2_cls', 1.0))
        self.stage2_bbox_weight = float(loss_weights.get('stage2_bbox', 1.0))

    def _sync_and_log(self, loss_dict, prefix):
        synced = OrderedDict()
        for loss_name, loss_value in loss_dict.items():
            if isinstance(loss_value, torch.Tensor):
                reduced_value = loss_value
                if dist.is_available() and dist.is_initialized():
                    reduced_value = reduced_value.data.clone()
                    dist.all_reduce(reduced_value.div_(dist.get_world_size()))
                synced[loss_name] = reduced_value.item()
                self.log(
                    f'{prefix}/{loss_name}',
                    reduced_value,
                    batch_size=1,
                    sync_dist=prefix != 'train',
                )
            else:
                synced[loss_name] = loss_value
        return synced

    def _forward_stage1(self, pts_data, imgs, metas):
        if imgs is None:
            raise ValueError('CVFusion expects images in the batch.')

        voxel_dict = self.voxelize(pts_data)
        voxels = voxel_dict['voxels']
        num_points = voxel_dict['num_points']
        coors = voxel_dict['coors']

        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = int(coors[-1, 0].item()) + 1
        radar_bev = self.middle_encoder(voxel_features, coors, batch_size)
        radar_feats = self.backbone(radar_bev)

        image_feats = self.image_backbone(imgs)
        target_shapes = [feat.shape[-2:] for feat in radar_feats]
        camera_bev_feats = self.bev_lifter(image_feats, metas, target_shapes)
        fused_feats, occupancy_maps = self.rg_iter_fusion(radar_feats, camera_bev_feats)
        neck_feats = self.neck(fused_feats)
        stage1_preds = self.head(neck_feats)
        return dict(
            stage1_preds=stage1_preds,
            image_feats=image_feats,
            fused_feats=fused_feats,
            neck_feats=neck_feats,
            occupancy_maps=occupancy_maps,
        )

    def _decode_stage1(self, stage1_preds, metas):
        with torch.no_grad():
            return self.head.get_bboxes(stage1_preds, img_metas=metas)

    def _forward_cvfusion(self, pts_data, imgs, metas, gt_bboxes_3d=None, gt_labels_3d=None, training=False):
        pts_with_flag = self._append_current_sweep_flag(pts_data)
        features = self._forward_stage1(pts_with_flag, imgs, metas)
        stage1_dets = self._decode_stage1(features['stage1_preds'], metas)
        proposals = self.proposal_refiner.build_proposals(
            stage1_dets,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            training=training,
        )
        refine_outputs = self.proposal_refiner(
            proposals,
            pts_with_flag,
            features['image_feats'],
            features['neck_feats'][0],
            metas,
        )
        refined_predictions = self.proposal_refiner.predict(refine_outputs, proposals)
        refined_bbox_list = self.proposal_refiner.postprocess(
            refined_predictions,
            min_radius=self.head.test_cfg['min_radius'],
            score_threshold=self.final_score_threshold,
            post_max_size=self.head.test_cfg['post_max_size'],
            nms_thr=self.head.test_cfg.get('nms_thr', 0.01),
        )
        return features, stage1_dets, proposals, refine_outputs, refined_bbox_list

    def training_step(self, batch, batch_idx):
        pts_data = batch['pts']
        imgs = batch.get('imgs')
        metas = batch['metas']
        gt_labels_3d = batch['gt_labels_3d']
        gt_bboxes_3d = batch['gt_bboxes_3d']

        features, _, proposals, refine_outputs, _ = self._forward_cvfusion(
            pts_data,
            imgs,
            metas,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            training=True,
        )

        stage1_losses = self.head.loss(gt_bboxes_3d, gt_labels_3d, features['stage1_preds'])
        stage2_losses = self.proposal_refiner.loss(refine_outputs, proposals, gt_bboxes_3d)

        log_vars = OrderedDict()
        for loss_name, loss_value in stage1_losses.items():
            log_vars[loss_name] = loss_value.mean()
        for loss_name, loss_value in stage2_losses.items():
            log_vars[loss_name] = loss_value.mean() if isinstance(loss_value, torch.Tensor) else loss_value

        total_loss = self.stage1_loss_weight * sum(
            value for key, value in log_vars.items()
            if isinstance(value, torch.Tensor) and key.startswith('task') and 'loss' in key
        )
        total_loss = total_loss + self.stage2_cls_weight * log_vars['stage2.loss_cls']
        total_loss = total_loss + self.stage2_bbox_weight * log_vars['stage2.loss_bbox']
        log_vars['loss'] = total_loss
        self._sync_and_log(log_vars, prefix='train')
        return total_loss

    def validation_step(self, batch, batch_idx):
        assert len(batch['pts']) == 1, 'Batch size should be 1 for validation'
        pts_data = batch['pts']
        imgs = batch.get('imgs')
        metas = batch['metas']
        gt_labels_3d = batch['gt_labels_3d']
        gt_bboxes_3d = batch['gt_bboxes_3d']

        features, _, proposals, refine_outputs, bbox_list = self._forward_cvfusion(
            pts_data,
            imgs,
            metas,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            training=False,
        )

        bbox_results = [
            dict(bboxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
            for bboxes, scores, labels in bbox_list
        ]

        stage1_losses = self.head.loss(gt_bboxes_3d, gt_labels_3d, features['stage1_preds'])
        stage2_losses = self.proposal_refiner.loss(refine_outputs, proposals, gt_bboxes_3d)
        log_vars = OrderedDict()
        for loss_name, loss_value in stage1_losses.items():
            log_vars[loss_name] = loss_value.mean()
        for loss_name, loss_value in stage2_losses.items():
            log_vars[loss_name] = loss_value.mean() if isinstance(loss_value, torch.Tensor) else loss_value

        total_loss = self.stage1_loss_weight * sum(
            value for key, value in log_vars.items()
            if isinstance(value, torch.Tensor) and key.startswith('task') and 'loss' in key
        )
        total_loss = total_loss + self.stage2_cls_weight * log_vars['stage2.loss_cls']
        total_loss = total_loss + self.stage2_bbox_weight * log_vars['stage2.loss_bbox']
        log_vars['loss'] = total_loss
        self._sync_and_log(log_vars, prefix='validation')

        self.val_results_list.append(dict(
            sample_idx=batch['metas'][0]['num_frame'],
            input_batch={'metas': [batch['metas'][0]]},
            bbox_results=bbox_results,
            losses={key: value.item() if isinstance(value, torch.Tensor) else value for key, value in log_vars.items()},
        ))

    def convert_valid_bboxes(self, box_dict, input_batch):
        box_preds = box_dict['bboxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = input_batch['metas'][0]['num_frame']

        vod_frame_data = FrameDataLoader(kitti_locations=self.vod_kitti_locations, frame_number=sample_idx)
        local_transforms = FrameTransformMatrix(vod_frame_data)

        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)
        box_preds = type(box_preds)(
            box_preds.tensor.detach().cpu(),
            box_dim=box_preds.box_dim,
            with_yaw=box_preds.with_yaw,
        )
        scores = scores.cpu()
        labels = labels.cpu()
        meta = input_batch['metas'][0]
        img_shape = meta.get('ori_img_shape')
        if img_shape is None:
            img_shape = meta.get('img_shape')
        if isinstance(img_shape, torch.Tensor):
            img_h, img_w = img_shape.tolist()
        else:
            img_h, img_w = (1216, 1936)

        corners_img = []
        centers_cam = []
        for box_corners, box_center in zip(box_preds.corners, box_preds.bottom_center):
            corners_homo = np.ones((8, 4), dtype=np.float32)
            corners_homo[:, :3] = box_corners.detach().cpu().numpy()
            corners_cam = homogeneous_transformation(corners_homo, local_transforms.t_camera_lidar)
            corners_proj = np.dot(corners_cam, local_transforms.camera_projection_matrix.T)
            corners_proj = (corners_proj[:, :2].T / np.clip(corners_proj[:, 2], a_min=1e-5, a_max=None)).T
            corners_proj = torch.from_numpy(corners_proj).float()
            corners_img.append(corners_proj)

            center_homo = np.ones((1, 4), dtype=np.float32)
            center_homo[:, :3] = box_center.detach().cpu().numpy()
            center_cam = homogeneous_transformation(center_homo, local_transforms.t_camera_lidar)
            centers_cam.append(torch.from_numpy(center_cam[:, :3]).float())

        if not corners_img:
            return dict(
                box2d=np.zeros([0, 4]),
                location_cam=np.zeros([0, 3]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx,
            )

        corners_img = torch.stack(corners_img, dim=0)
        centers_cam = torch.cat(centers_cam, dim=0)
        minxy = torch.min(corners_img, dim=1)[0]
        maxxy = torch.max(corners_img, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)

        pc_range = self.pc_range.cpu()
        valid_cam = (
            (box_2d_preds[:, 0] < img_w)
            & (box_2d_preds[:, 1] < img_h)
            & (box_2d_preds[:, 2] > 0)
            & (box_2d_preds[:, 3] > 0)
        )
        valid_pcd = ((box_preds.center > pc_range[:3]) & (box_preds.center < pc_range[3:])).all(dim=-1)
        valid = valid_cam & valid_pcd
        has_valid = bool(valid.any().item())
        return dict(
            box2d=box_2d_preds[valid].cpu().numpy() if has_valid else np.zeros([0, 4]),
            location_cam=centers_cam[valid].cpu().numpy() if has_valid else np.zeros([0, 3]),
            box3d_lidar=box_preds[valid].tensor.cpu().numpy() if has_valid else np.zeros([0, 7]),
            scores=scores[valid].cpu().numpy() if has_valid else np.zeros([0]),
            label_preds=labels[valid].cpu().numpy() if has_valid else np.zeros([0, 4]),
            sample_idx=sample_idx,
        )
