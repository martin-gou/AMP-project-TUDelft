import torch

def collate_vod_batch(batch):
    pts_list = []
    img_list = []
    gt_labels_3d_list = []
    gt_bboxes_3d_list = []
    meta_list = []
    has_images = batch[0].get('image') is not None
    for idx, sample in enumerate(batch):
        pts_list.append(sample['lidar_data'])
        if has_images:
            img_list.append(sample['image'])
        gt_labels_3d_list.append(sample['gt_labels_3d'])
        gt_bboxes_3d_list.append(sample['gt_bboxes_3d'])
        meta_list.append(sample['meta'])
    return dict(
        pts = pts_list,
        imgs = torch.stack(img_list, dim=0) if has_images else None,
        gt_labels_3d = gt_labels_3d_list,
        gt_bboxes_3d = gt_bboxes_3d_list,
        metas = meta_list
    )
    
