# Training Report

| Timestamp | Exp ID | Run Name | Model | Epochs | Batch | LR | WD | Val Every | Best ROI mAP | Best ROI Model | Best Entire mAP | Best Entire Model | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260323_224221 | centerpoint_baseline_db_try_slurm | centerpoint_baseline_db_try_slurm_20260323_224221 | centerpoint_radar | 30 | 4 | 0.001 | 0.01 | 3 | 55.268536 | `outputs/centerpoint_baseline_db_try_slurm/20260323_224221/checkpoints/roi-ep11-centerpoint_baseline_db_try_slurm.ckpt` | 32.745811 | `outputs/centerpoint_baseline_db_try_slurm/20260323_224221/checkpoints/entire-ep2-centerpoint_baseline_db_try_slurm.ckpt` | `outputs/centerpoint_baseline_db_try_slurm/20260323_224221` |
| 20260324_001548 | radar_pfn2 | radar_pfn2_20260324_001548 | centerpoint_radar_pfn2 | 30 | 4 | 0.001 | 0.01 | 3 | 55.390411 | `outputs/radar_pfn2/20260324_001548/checkpoints/roi-ep5-radar_pfn2.ckpt` | 34.445076 | `outputs/radar_pfn2/20260324_001548/checkpoints/entire-ep2-radar_pfn2.ckpt` | `outputs/radar_pfn2/20260324_001548` |
| 20260324_021311 | radar_pfn2_gate | radar_pfn2_gate_20260324_021311 | centerpoint_radar_pfn2_gate | 30 | 4 | 0.001 | 0.01 | 3 | 58.503181 | `outputs/radar_pfn2_gate/20260324_021311/checkpoints/roi-ep8-radar_pfn2_gate.ckpt` | 34.498543 | `outputs/radar_pfn2_gate/20260324_021311/checkpoints/entire-ep2-radar_pfn2_gate.ckpt` | `outputs/radar_pfn2_gate/20260324_021311` |
| 20260324_040937 | radar_pfn2_gate_e4 | radar_pfn2_gate_e4_20260324_040937 | centerpoint_radar_pfn2_gate_e4 | 30 | 4 | 0.001 | 0.01 | 3 | 56.366013 | `outputs/radar_pfn2_gate_e4/20260324_040937/checkpoints/roi-ep5-radar_pfn2_gate_e4.ckpt` | 34.289917 | `outputs/radar_pfn2_gate_e4/20260324_040937/checkpoints/entire-ep5-radar_pfn2_gate_e4.ckpt` | `outputs/radar_pfn2_gate_e4/20260324_040937` |
| 20260324_061344 | radar_pfn2_gate_e5 | radar_pfn2_gate_e5_20260324_061344 | centerpoint_radar_pfn2_gate_e5 | 30 | 4 | 0.001 | 0.01 | 3 | 58.186096 | `outputs/radar_pfn2_gate_e5/20260324_061344/checkpoints/roi-ep14-radar_pfn2_gate_e5.ckpt` | 33.427948 | `outputs/radar_pfn2_gate_e5/20260324_061344/checkpoints/entire-ep5-radar_pfn2_gate_e5.ckpt` | `outputs/radar_pfn2_gate_e5/20260324_061344` |
| 20260324_122514 | radar_pfn2_gate_e7 | radar_pfn2_gate_e7_20260324_122514 | centerpoint_radar_pfn2_gate_e7 | 20 | 4 | 0.001 | 0.01 | 3 | 58.667835 | `outputs/radar_pfn2_gate_e7/20260324_122514/checkpoints/roi-ep2-radar_pfn2_gate_e7.ckpt` | 34.349380 | `outputs/radar_pfn2_gate_e7/20260324_122514/checkpoints/entire-ep2-radar_pfn2_gate_e7.ckpt` | `outputs/radar_pfn2_gate_e7/20260324_122514` |
| 20260325_002204 | radar_pfn2_gate_e8 | radar_pfn2_gate_e8_20260325_002204 | centerpoint_radar_pfn2_gate_e8 | 40 | 4 | 0.001 | 0.01 | 3 | 56.760063 | `outputs/radar_pfn2_gate_e8/20260325_002204/checkpoints/roi-ep14-radar_pfn2_gate_e8.ckpt` | 33.380836 | `outputs/radar_pfn2_gate_e8/20260325_002204/checkpoints/entire-ep17-radar_pfn2_gate_e8.ckpt` | `outputs/radar_pfn2_gate_e8/20260325_002204` |
| 20260325_042113 | radar_pfn2_gate_e9 | radar_pfn2_gate_e9_20260325_042113 | centerpoint_radar_pfn2_gate_e9 | 40 | 4 | 0.0005 | 0.01 | 3 | 55.742443 | `outputs/radar_pfn2_gate_e9/20260325_042113/checkpoints/roi-ep29-radar_pfn2_gate_e9.ckpt` | 31.261379 | `outputs/radar_pfn2_gate_e9/20260325_042113/checkpoints/entire-ep14-radar_pfn2_gate_e9.ckpt` | `outputs/radar_pfn2_gate_e9/20260325_042113` |

<!-- REPORT_ANALYSIS_START -->

## E7-E9 Analysis

Summary:
- E7 reached the best ROI mAP among the camera-phase runs, but it did not actually use camera features for detection decisions. It is therefore best interpreted as a control run that validates image loading and feature extraction overhead rather than a successful fusion model.
- E8 reduced ROI mAP from 58.67 to 56.76 and entire-area mAP from 34.35 to 33.38.
- E9 reduced performance further to 55.74 ROI mAP and 31.26 entire-area mAP.

Interpretation:
- The current point-level camera fusion design adds noisy image information to sparse radar points and does not improve the radar representation.
- The image branch uses late ResNet-18 features at low spatial resolution, so point-wise feature sampling is coarse and many projected radar points receive weak or overly similar visual descriptors.
- The frozen ImageNet-pretrained visual features are not well aligned with VoD radar detection, so direct concatenation before voxelization likely perturbs a radar backbone that is already strong on its own.
- In E9, partial fine-tuning still failed to recover the drop, which suggests the issue is not just lack of adaptation but also the current fusion location and supervision path.

Conclusion:
- E8 and E9 should be treated as negative results for the current camera fusion design.
- The strongest model in this sequence remains the radar-only E7 control / E5-class radar backbone family, not the camera-fused variants.
- For future work, a better direction would be to fuse image information at a differentiable BEV stage or with stronger alignment than sparse pre-voxel point sampling.
| 20260325_113618 | radar_pfn2_gate_e7 | radar_pfn2_gate_e7_20260325_113618 | centerpoint_radar_pfn2_gate_e7 | 40 | 4 | 0.001 | 0.01 | 3 | 58.709103 | `outputs/radar_pfn2_gate_e7/20260325_113618/checkpoints/roi-ep29-radar_pfn2_gate_e7.ckpt` | 34.349380 | `outputs/radar_pfn2_gate_e7/20260325_113618/checkpoints/entire-ep2-radar_pfn2_gate_e7.ckpt` | `outputs/radar_pfn2_gate_e7/20260325_113618` |
