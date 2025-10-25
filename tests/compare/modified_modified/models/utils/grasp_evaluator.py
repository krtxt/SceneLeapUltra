from statistics import mean

import numpy as np
import torch

from utils.evaluate_utils import cal_pen, cal_q1


class GraspMetricCalculator:
    def __init__(self, q1_cfg, hand_model, scale=0.1):
        self.q1_cfg = q1_cfg
        self.hand_model = hand_model
        self.scale = scale

    def calculate_metrics(self, pred_dict, batch):
        """
        Calculate evaluation metrics supporting both single and multi-grasp formats.

        For multi-grasp format, computes metrics for all grasps and provides both
        per-grasp and aggregated statistics.
        """
        hand_model_pose = pred_dict['matched']['hand_model_pose']

        # Detect input format
        if hand_model_pose.dim() == 2:
            # Single grasp format: [B, pose_dim]
            return self._calculate_metrics_single_grasp(pred_dict, batch)
        elif hand_model_pose.dim() == 3:
            # Multi grasp format: [B, num_grasps, pose_dim]
            return self._calculate_metrics_multi_grasp(pred_dict, batch)
        else:
            raise ValueError(f"Unsupported hand_model_pose dimension: {hand_model_pose.dim()}")

    def _calculate_metrics_single_grasp(self, pred_dict, batch):
        """Calculate metrics for single grasp format (backward compatibility)"""
        q1_list = []
        pen_list = []
        valid_q1_list = []
        metric_details = {}

        hand_model_pose = pred_dict['matched']['hand_model_pose']
        obj_verts = pred_dict['matched']['obj_verts']
        obj_faces = pred_dict['matched']['obj_faces']

        for i in range(hand_model_pose.shape[0]):
            q1 = cal_q1(self.q1_cfg, self.hand_model,
                       obj_verts[i],
                       obj_faces[i],
                       self.scale,
                       hand_model_pose[i].unsqueeze(0))

            pen = cal_pen(self.q1_cfg, self.hand_model,
                         obj_verts[i],
                         obj_faces[i],
                         self.scale,
                         hand_model_pose[i].unsqueeze(0))

            # Convert tensor results to float if needed
            q1_val = q1.item() if torch.is_tensor(q1) else q1
            pen_val = pen.item() if torch.is_tensor(pen) else pen

            q1_list.append(q1_val)
            pen_list.append(pen_val)
            thres_pen = self.q1_cfg['thres_pen']
            valid = (pen_val < thres_pen)
            valid_q1 = q1_val if valid else 0
            valid_q1_list.append(valid_q1)

            metric_details[f"{batch['obj_code'][i]}_{batch['scene_id'][i]}_{batch['category_id_from_object_index'][i]}_{batch['depth_view_index'][i]}"] = {
                "q1": q1_val,
                "valid_q1": valid_q1,
                "pen": pen_val
            }

        metric_dict = {
            "mean_q1": float(mean(q1_list)),
            "mean_pen": float(mean(pen_list)),
            "max_pen": float(max(pen_list)),
            "mean_valid_q1": float(mean(valid_q1_list)),
        }
        return metric_dict, metric_details

    def _calculate_metrics_multi_grasp(self, pred_dict, batch):
        """Calculate metrics for multi-grasp format with batch processing"""
        hand_model_pose = pred_dict['matched']['hand_model_pose']  # [B, num_grasps, pose_dim]
        obj_verts = pred_dict['matched']['obj_verts']  # [B, V, 3]
        obj_faces = pred_dict['matched']['obj_faces']  # [B, F, 3]

        B, num_grasps, pose_dim = hand_model_pose.shape

        # Batch process all samples
        q1_results = []
        pen_results = []
        metric_details = {}

        for i in range(B):
            # Calculate Q1 and penetration for all grasps of current sample
            q1_batch = cal_q1(self.q1_cfg, self.hand_model,
                             obj_verts[i],
                             obj_faces[i],
                             self.scale,
                             hand_model_pose[i])  # [num_grasps]

            pen_batch = cal_pen(self.q1_cfg, self.hand_model,
                               obj_verts[i],
                               obj_faces[i],
                               self.scale,
                               hand_model_pose[i])  # [num_grasps]

            # Convert to numpy for easier processing
            q1_vals = q1_batch.cpu().numpy() if torch.is_tensor(q1_batch) else np.array([q1_batch])
            pen_vals = pen_batch.cpu().numpy() if torch.is_tensor(pen_batch) else np.array([pen_batch])

            q1_results.extend(q1_vals.tolist())
            pen_results.extend(pen_vals.tolist())

            # Store per-grasp details
            for j in range(num_grasps):
                q1_val = q1_vals[j] if len(q1_vals) > j else q1_vals[0]
                pen_val = pen_vals[j] if len(pen_vals) > j else pen_vals[0]
                thres_pen = self.q1_cfg['thres_pen']
                valid = (pen_val < thres_pen)
                valid_q1 = q1_val if valid else 0

                key = f"{batch['obj_code'][i]}_{batch['scene_id'][i]}_{batch['category_id_from_object_index'][i]}_{batch['depth_view_index'][i]}_grasp{j}"
                metric_details[key] = {
                    "q1": float(q1_val),
                    "valid_q1": float(valid_q1),
                    "pen": float(pen_val)
                }

        # Calculate aggregated metrics
        valid_q1_list = [details["valid_q1"] for details in metric_details.values()]

        # Multi-grasp specific metrics
        q1_array = np.array(q1_results)
        pen_array = np.array(pen_results)

        # Best grasp metrics (minimum penetration, maximum Q1 among valid grasps)
        best_grasp_indices = []
        best_q1_list = []
        best_pen_list = []

        for i in range(B):
            start_idx = i * num_grasps
            end_idx = (i + 1) * num_grasps
            sample_q1 = q1_array[start_idx:end_idx]
            sample_pen = pen_array[start_idx:end_idx]

            thres_pen = self.q1_cfg['thres_pen']
            valid_mask = sample_pen < thres_pen
            if valid_mask.any():
                valid_indices = np.where(valid_mask)[0]
                best_idx = valid_indices[np.argmax(sample_q1[valid_indices])]
            else:
                best_idx = np.argmin(sample_pen)

            best_grasp_indices.append(best_idx)
            best_q1_list.append(sample_q1[best_idx])
            best_pen_list.append(sample_pen[best_idx])

        metric_dict = {
            # Overall metrics (all grasps)
            "mean_q1": float(np.mean(q1_results)),
            "mean_pen": float(np.mean(pen_results)),
            "max_pen": float(np.max(pen_results)),
            "mean_valid_q1": float(mean(valid_q1_list)),

            # Multi-grasp specific metrics
            "std_q1": float(np.std(q1_results)),
            "std_pen": float(np.std(pen_results)),
            "min_q1": float(np.min(q1_results)),
            "max_q1": float(np.max(q1_results)),
            "min_pen": float(np.min(pen_results)),

            # Best grasp metrics
            "best_mean_q1": float(np.mean(best_q1_list)),
            "best_mean_pen": float(np.mean(best_pen_list)),
            "best_max_pen": float(np.max(best_pen_list)),
            "best_mean_valid_q1": float(np.mean([q1 if pen < thres_pen else 0
                                               for q1, pen in zip(best_q1_list, best_pen_list)])),

            # Success rate metrics
            "success_rate": float(np.mean(pen_array < thres_pen)),
            "best_success_rate": float(np.mean(np.array(best_pen_list) < thres_pen)),
        }

        return metric_dict, metric_details 