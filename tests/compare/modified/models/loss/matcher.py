from functools import partial
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class Matcher(nn.Module):
    def __init__(self, weight_dict, rot_type):
        super().__init__()
        self.weight_dict = {k: v for k, v in weight_dict.items() if v > 0}
        self.rot_type = rot_type

    @torch.no_grad()
    def forward(self, preds, targets):
        """
        Performs optimal matching between multiple predictions and multiple targets.

        The matching process consists of three main steps:
        1.  Compute a cost matrix between every prediction and every target. The cost
            represents the "dissimilarity" between a prediction-target pair.
            The shape of this matrix for each batch item is [num_grasps, num_targets].
        2.  Use the Hungarian algorithm (scipy.optimize.linear_sum_assignment) to find
            the optimal one-to-one assignment that minimizes the total cost.
        3.  Return the assignment information, including indices and masks.

        Args:
            preds (dict): A dictionary containing model predictions.
                - 'pred_pose_norm': torch.Tensor of shape [B, num_grasps, 22]
                - ... and its sliced components ('translation_norm', 'qpos_norm', 'rotation').
            targets (dict): A dictionary containing ground truth data.
                - 'norm_pose': torch.Tensor of shape [B, num_targets, 22]

        Returns:
            A dictionary containing the following keys:
            - 'final_cost' (np.array): The combined cost matrix of shape [B, num_grasps, num_targets].
            - 'assignments' (list): The raw output from linear_sum_assignment for each batch item.
            - 'per_query_gt_inds' (torch.Tensor): For each prediction, the index of the matched
              target. Shape: [B, num_grasps].
            - 'query_matched_mask' (torch.Tensor): A binary mask indicating which predictions
              were successfully matched. Shape: [B, num_grasps].
        """
        device = preds["rotation"].device

        # Get dimensional information
        if preds["rotation"].dim() == 3:
            # Multi-grasp mode: [B, num_grasps, rot_dim]
            batch_size, nqueries = preds["rotation"].shape[:2]
        else:
            # Single-grasp mode: [B, rot_dim]
            batch_size = preds["rotation"].shape[0]
            nqueries = 1
            # Expand dimensions for unified processing
            for key in preds:
                if isinstance(preds[key], torch.Tensor) and preds[key].dim() == 2:
                    preds[key] = preds[key].unsqueeze(1)  # [B, 1, D]

        # Calculate valid target mask and check assumption
        valid_mask = (targets["norm_pose"].abs().sum(dim=-1) > 0)
        num_valid_targets_per_sample = valid_mask.sum(dim=1)
        if (num_valid_targets_per_sample < nqueries).any():
            raise ValueError(
                f"Matcher assumption violated: The number of valid ground truth grasps must be >= "
                f"the number of predicted grasps for every sample in the batch. "
                f"Found a sample with {num_valid_targets_per_sample.min().item()} targets for {nqueries} queries."
            )

        # Initialize cost matrix. Shape: [num_cost_terms, B, num_grasps, num_targets]
        cost_matrices = torch.zeros(
            len(self.weight_dict),
            batch_size,
            nqueries,
            targets["norm_pose"].size(1),
            device=device
        )

        # Calculate individual costs
        for i, (name, weight) in enumerate(self.weight_dict.items()):
            m = getattr(self, f"get_{name}_cost_mat")
            if name == "rotation":
                cost_matrices[i] = m(
                    preds, targets, weight, self.rot_type, valid_mask
                )
            else:
                cost_matrices[i] = m(
                    preds, targets, weight, valid_mask
                )

        # Sum up cost matrices to get the final cost. Shape: [B, num_grasps, num_targets]
        final_cost = cost_matrices.sum(0).cpu().numpy()

        # Solve using Hungarian algorithm for each item in the batch.
        assignments = []
        per_query_gt_inds = torch.zeros(
            [batch_size, nqueries], dtype=torch.int64, device=device
        )
        query_matched_mask = torch.zeros(
            [batch_size, nqueries], dtype=torch.int64, device=device
        )

        for b in range(batch_size):
            valid_targets = valid_mask[b].sum().item()
            if valid_targets > 0:
                # Many-to-many matching: nqueries predictions vs valid_targets targets
                assign = linear_sum_assignment(final_cost[b, :, :valid_targets])
                assign = [torch.from_numpy(x).long().to(device) for x in assign]
                per_query_gt_inds[b, assign[0]] = assign[1]
                query_matched_mask[b, assign[0]] = 1
                assignments.append(assign)
            else:
                assignments.append([
                    torch.tensor([], device=device),
                    torch.tensor([], device=device)
                ])

        return {
            "final_cost": final_cost,
            "assignments": assignments,
            "per_query_gt_inds": per_query_gt_inds,
            "query_matched_mask": query_matched_mask,
            "is_multi_grasp": preds["rotation"].dim() == 3,
            "nqueries": nqueries,
            "num_targets": targets["norm_pose"].size(1)
        }

    def get_qpos_cost_mat(self, prediction, targets, weight=1.0, valid_mask=None):
        return self._get_cost_mat_by_elementwise(
            prediction["qpos_norm"], 
            targets["norm_pose"][..., 3:19], 
            weight, 
            valid_mask
        )

    def get_translation_cost_mat(self, prediction, targets, weight=1.0, valid_mask=None):
        return self._get_cost_mat_by_elementwise(
            prediction["translation_norm"], 
            targets["norm_pose"][..., :3], 
            weight, 
            valid_mask
        )

    def get_rotation_cost_mat(
        self, prediction, targets, weight=1.0, rotation_type="euler", valid_mask=None
    ):
        m = getattr(self, f"_get_{rotation_type}_cost_mat", None)
        if m:
            return m(
                prediction["rotation"], 
                targets["norm_pose"][..., 19:], 
                weight, 
                valid_mask
            )
        raise NotImplementedError(f"Rotation type {rotation_type} not supported")

    def _get_cost_mat_by_elementwise(
        self,
        prediction,
        targets,
        weight=1.0,
        valid_mask=None,
        element_wise_func=partial(F.l1_loss, reduction="none")
    ):
        """
        Generic cost matrix calculation, supports multiple prediction inputs.
        This function computes the cost between each prediction and each target using
        element-wise L1 loss.

        It uses broadcasting to efficiently compute the pairwise costs.
        - Prediction shape: [B, nqueries, D] -> [B, nqueries, 1, D]
        - Target shape:     [B, N, D]       -> [B, 1, N, D]
        The result is a cost matrix of shape [B, nqueries, N].
        """

        # Process input dimensions
        if prediction.dim() == 2:
            # Single prediction mode: [B, D] -> [B, 1, D]
            prediction = prediction.unsqueeze(1)

        B, nqueries, D = prediction.shape
        N = targets.shape[1]

        # Broadcast to calculate cost matrix
        cost = element_wise_func(
            prediction.unsqueeze(2).expand(-1, -1, N, -1),  # [B, nqueries, N, D]
            targets.unsqueeze(1).expand(-1, nqueries, -1, -1)  # [B, nqueries, N, D]
        ).sum(-1)  # [B, nqueries, N]

        # Apply validity mask
        if valid_mask is not None:
            cost = cost.masked_fill(~valid_mask.unsqueeze(1), float('inf'))

        return weight * cost

    def _get_quat_cost_mat(self, prediction, targets, weight=1.0, valid_mask=None):
        """
        Quaternion cost matrix calculation, supports multiple prediction inputs.
        Cost is calculated as 1 - |dot(p, q)|, which measures the angular distance.
        """

        # Process input dimensions
        if prediction.dim() == 2:
            # Single prediction mode: [B, D] -> [B, 1, D]
            prediction = prediction.unsqueeze(1)

        # Calculate quaternion similarity cost
        cost = 1 - torch.matmul(prediction, targets.transpose(-2, -1)).abs()

        if valid_mask is not None:
            cost = cost.masked_fill(~valid_mask.unsqueeze(1), float('inf'))
        return weight * cost

    def _get_r6d_cost_mat(self, prediction, targets, weight=1.0, valid_mask=None):
        return self._get_cost_mat_by_elementwise(
            prediction, 
            targets, 
            weight, 
            valid_mask
        )

    def _get_euler_cost_mat(self, prediction, targets, weight=1.0, valid_mask=None):
        # TODO: This function has a known bug in its periodicity calculation.
        # AGENT_NOTE: Do not implement this function. It is not used in the current project setup.
        raise NotImplementedError("Euler angle cost is not correctly implemented and should not be used.")
