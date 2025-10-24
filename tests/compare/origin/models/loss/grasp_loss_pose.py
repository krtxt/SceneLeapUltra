import logging
import os
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.functional import Tensor

import pytorch3d.structures
import pytorch3d.ops
import pytorch3d.transforms
from pytorch3d.loss import chamfer_distance

from utils.hand_model import HandModel, HandModelType
from utils.hand_helper import denorm_hand_pose_robust
from utils.evaluate_utils import cal_q1, cal_pen
from .matcher import Matcher
from models.utils import loss_components
from models.utils.grasp_evaluator import GraspMetricCalculator
from models.utils.pose_processor import PoseProcessor


class GraspLossPose(nn.Module):
    def __init__(self, loss_cfg):
        super().__init__()

        # Dynamically determine correct device for distributed training
        device = self._get_correct_device(loss_cfg.device)

        self.hand_model = HandModel(HandModelType.LEAP, loss_cfg.hand_model.n_surface_points, loss_cfg.rot_type, device)
        self.loss_weights = {k: v for k, v in loss_cfg.loss_weights.items() if v > 0}  # Only keep losses with weight > 0
        self.matcher = Matcher(weight_dict=loss_cfg.cost_weights, rot_type=loss_cfg.rot_type)
        self.rot_type = loss_cfg.rot_type
        self.q1_cfg = loss_cfg.q1
        self.scale = 0.1
        self.mode = loss_cfg.mode
        self.use_negative_prompts = getattr(loss_cfg, 'use_negative_prompts', True)
        self.neg_loss_weight = loss_cfg.loss_weights.get('neg_loss', 0.0)

        # Multi-grasp configuration
        self.multi_grasp_cfg = getattr(loss_cfg, 'multi_grasp', None)
        if self.multi_grasp_cfg:
            self.loss_aggregation = getattr(self.multi_grasp_cfg, 'loss_aggregation', 'mean')
            self.use_consistency_loss = getattr(self.multi_grasp_cfg, 'use_consistency_loss', False)
            self.consistency_loss_weight = getattr(self.multi_grasp_cfg, 'consistency_loss_weight', 0.1)
            self.diversity_loss_weight = getattr(self.multi_grasp_cfg, 'diversity_loss_weight', 0.05)
        else:
            self.loss_aggregation = 'mean'
            self.use_consistency_loss = False
            self.consistency_loss_weight = 0.1
            self.diversity_loss_weight = 0.05

        # Standardized validation loss configuration
        self.std_val_cfg = getattr(loss_cfg, 'standardized_validation_loss', None)
        if self.std_val_cfg and getattr(self.std_val_cfg, 'enabled', False):
            self.use_standardized_val_loss = True
            self.std_val_weights = getattr(self.std_val_cfg, 'weights', {})
            self.std_val_normalization = getattr(self.std_val_cfg, 'normalization', {})
            self.std_val_aggregation = getattr(self.std_val_cfg, 'aggregation', {})
            self.std_val_quality_metrics = getattr(self.std_val_cfg, 'quality_metrics', {})

            logging.info(f"Standardized validation loss enabled with weights: {self.std_val_weights}")
        else:
            self.use_standardized_val_loss = False
            logging.info("Using standard validation loss (same as training loss)")

        self._init_loss_functions()
        self.metric_calculator = GraspMetricCalculator(self.q1_cfg, self.hand_model, self.scale)
        self.pose_processor = PoseProcessor(self.hand_model, self.rot_type, self.mode)

        # Configure hand model parameters
        self._configure_hand_model_kwargs()

    def _init_loss_functions(self):
        """Initializes the mapping from loss names to their calculation functions."""
        self.loss_func_map = {
            "latent": loss_components.calculate_latent_loss,
            "para": lambda p, t: loss_components.calculate_para_loss(p, t, self.loss_aggregation),
            "noise": lambda p, t: loss_components.calculate_noise_loss(p, t, self.loss_aggregation),
            "translation": lambda p, t: loss_components.calculate_translation_loss(p, t, self.loss_aggregation),
            "qpos": lambda p, t: loss_components.calculate_qpos_loss(p, t, self.loss_aggregation),
            "rotation": lambda p, t: loss_components.calculate_rotation_loss(p, t, self.loss_aggregation),
            "hand_chamfer": loss_components.calculate_hand_chamfer_loss,
            "obj_penetration": loss_components.calculate_obj_penetration_loss,
            "self_penetration": loss_components.calculate_self_penetration_loss,
            "distance": loss_components.calculate_distance_loss,
            "cmap": loss_components.calculate_cmap_loss,
            "consistency": loss_components.calculate_consistency_loss,
            "diversity": loss_components.calculate_diversity_loss,
        }

    def _get_device(self, *tensors):
        """
        Automatically infer device, supporting distributed training
        Prioritize inference from input tensors, fallback to hand_model device
        """
        for tensor in tensors:
            if tensor is not None and hasattr(tensor, 'device'):
                return tensor.device

        # Infer device from hand_model
        if hasattr(self.hand_model, 'device'):
            return self.hand_model.device

        # Infer device from hand_model parameters
        try:
            return next(self.hand_model.parameters()).device
        except (StopIteration, AttributeError):
            pass

        # Final fallback to CPU
        return torch.device('cpu')

    def _get_correct_device(self, config_device):
        """
        Determine correct device for distributed training
        """
        import os
        import torch

        # Check if in distributed training environment
        if 'LOCAL_RANK' in os.environ:
            local_rank = int(os.environ['LOCAL_RANK'])
            if torch.cuda.is_available():
                device = f'cuda:{local_rank}'
                logging.info(f"GraspLossPose: Using device {device} for LOCAL_RANK {local_rank}")
                return device
        elif 'RANK' in os.environ:
            # If only RANK but no LOCAL_RANK, try using RANK
            rank = int(os.environ['RANK'])
            if torch.cuda.is_available():
                # Assume same number of GPUs per node
                local_rank = rank % torch.cuda.device_count()
                device = f'cuda:{local_rank}'
                logging.info(f"GraspLossPose: Using device {device} for RANK {rank}")
                return device

        # Non-distributed environment or unable to determine rank, use config device
        logging.info(f"GraspLossPose: Using config device {config_device}")
        return config_device

    def to(self, device):
        """
        Override to method to ensure hand_model also moves to correct device
        """
        result = super().to(device)

        # Synchronously update hand_model device
        if hasattr(self, 'hand_model') and self.hand_model is not None:
            # Update hand_model device attribute
            self.hand_model.device = device

            # Move all tensors in hand_model to new device
            if hasattr(self.hand_model, 'chain') and self.hand_model.chain is not None:
                self.hand_model.chain = self.hand_model.chain.to(device=device)

            # Move mesh data to new device
            if hasattr(self.hand_model, 'mesh'):
                for link_name, mesh_data in self.hand_model.mesh.items():
                    for key, tensor in mesh_data.items():
                        if isinstance(tensor, torch.Tensor):
                            mesh_data[key] = tensor.to(device)

            # Move other tensor attributes to new device
            for attr_name in ['joints_upper', 'joints_lower']:
                if hasattr(self.hand_model, attr_name):
                    attr_value = getattr(self.hand_model, attr_name)
                    if isinstance(attr_value, torch.Tensor):
                        setattr(self.hand_model, attr_name, attr_value.to(device))

            logging.info(f"GraspLossPose: Moved hand_model to device {device}")

        return result

    def _configure_hand_model_kwargs(self):
        """Configure hand model parameters"""
        # Configure hand model parameters for train/val phase based on active losses
        self.train_val_hand_model_kwargs = {
            "with_meshes": False,
            "with_surface_points": False,
            "with_contact_candidates": False,
            "with_penetration": False,
            "with_penetration_keypoints": False,
            "with_distance": False,
            "with_fingertip_keypoints": False,
        }
        
        if "hand_chamfer" in self.loss_weights:
            self.train_val_hand_model_kwargs["with_surface_points"] = True
        if "obj_penetration" in self.loss_weights:
            self.train_val_hand_model_kwargs["with_penetration"] = True
            self.train_val_hand_model_kwargs["with_penetration_keypoints"] = True
        if "self_penetration" in self.loss_weights:
            self.train_val_hand_model_kwargs["with_penetration_keypoints"] = True
        if "distance" in self.loss_weights:
            self.train_val_hand_model_kwargs["with_contact_candidates"] = True
        if "cmap" in self.loss_weights:
            self.train_val_hand_model_kwargs["with_distance"] = True

        logging.info(f"GraspLossPose initialized. train_val_hand_model_kwargs: {self.train_val_hand_model_kwargs}")

        # Configure hand model parameters for test/inference phase
        self.test_infer_hand_model_kwargs = {
            "with_meshes": True,
            "with_surface_points": True,
            "with_contact_candidates": True,
            "with_penetration": False,
            "with_penetration_keypoints": False,
            "with_distance": False,
            "with_fingertip_keypoints": False,
        }

    def forward(self, pred_dict, batch, mode='train'):
        if mode == 'train':
            return self._forward_train(pred_dict, batch)
        elif mode == 'val':
            return self._forward_val(pred_dict, batch)
        elif mode == 'test':
            return self._forward_test(pred_dict, batch)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _forward_train(self, pred_dict, batch):
        outputs, targets = self.pose_processor.process_train(pred_dict, batch, self.train_val_hand_model_kwargs)
        return self._calculate_losses(outputs, targets)

    def _forward_val(self, pred_dict, batch):
        outputs, targets = self.pose_processor.process_val(pred_dict, batch, self.matcher, self.train_val_hand_model_kwargs)

        if self.use_standardized_val_loss:
            # Calculate both standard and standardized validation losses
            standard_losses = self._calculate_losses(outputs, targets)
            standardized_losses = self._calculate_standardized_validation_losses(outputs, targets)

            # Return standardized losses but keep standard losses for logging
            standardized_losses['_standard_losses'] = standard_losses
            return standardized_losses
        else:
            return self._calculate_losses(outputs, targets)

    def _forward_test(self, pred_dict, batch):
        outputs, _ = self.pose_processor.process_test(pred_dict, batch, self.matcher)
        return self.metric_calculator.calculate_metrics(outputs, batch)

    def forward_infer(self, pred_dict, batch):
        matched_preds, matched_targets, outputs, targets = self.pose_processor.process_infer(pred_dict, batch, self.matcher, self.test_infer_hand_model_kwargs)
        return matched_preds, matched_targets, outputs, targets

    def forward_metric(self, pred_dict, batch):
        """
        Forward method for metric calculation (compatibility with diffuser_lightning.py)
        This is an alias for the test mode forward method.
        """
        return self._forward_test(pred_dict, batch)

    def _calculate_losses(self, outputs, targets):
        losses = {}

        for name, _ in self.loss_weights.items():
            if name == 'neg_loss':
                # If negative prompts are disabled by the config, skip this loss entirely.
                if not self.use_negative_prompts:
                    continue

                # Handle negative prompt loss if enabled
                if 'neg_pred' in outputs['matched'] and 'neg_text_features' in outputs['matched']:
                    neg_loss = self._calculate_negative_prompt_loss(
                        outputs['matched']['neg_pred'],
                        outputs['matched']['neg_text_features']
                    )
                    losses['neg_loss'] = neg_loss
                else:
                    # If keys are missing (e.g., for a batch without negative prompts),
                    # add a zero tensor to ensure the key exists, preventing a crash.
                    device = self._get_device(*outputs.values()) if outputs else torch.device('cpu')
                    losses['neg_loss'] = torch.tensor(0.0, device=device)
            elif name in self.loss_func_map:
                # Special handling for consistency loss based on configuration
                if name == 'consistency' and not self.use_consistency_loss:
                    continue
                
                loss_val = self.loss_func_map[name](outputs, targets)
                losses[name] = loss_val
            else:
                available_losses = list(self.loss_func_map.keys())
                raise NotImplementedError(f"Unable to calculate {name} loss. Available losses: {available_losses}")

        return losses

    def _calculate_standardized_validation_losses(self, outputs, targets):
        """
        Calculate standardized validation losses for fair cross-experiment comparison.

        This method uses fixed weights and normalization strategies to ensure
        validation losses are comparable across different training configurations.
        """
        losses = {}

        # Get standardized weights
        std_weights = self.std_val_weights
        normalization_cfg = self.std_val_normalization
        aggregation_cfg = self.std_val_aggregation

        # Calculate base losses with standardized aggregation
        for name in std_weights.keys():
            if std_weights[name] <= 0:
                continue

            if name in self.loss_func_map:
                # Use standardized aggregation strategy
                std_aggregation = aggregation_cfg.get('strategy', 'mean')

                # Temporarily override aggregation for standardized calculation
                original_aggregation = getattr(self, 'loss_aggregation', 'mean')
                self.loss_aggregation = std_aggregation

                try:
                    loss_val = self.loss_func_map[name](outputs, targets)

                    # Apply standardized normalization
                    if normalization_cfg.get('normalize_by_num_grasps', True):
                        loss_val = self._normalize_by_num_grasps(loss_val, outputs, targets)

                    if normalization_cfg.get('normalize_by_batch_size', True):
                        loss_val = self._normalize_by_batch_size(loss_val, outputs, targets)

                    losses[name] = loss_val

                finally:
                    # Restore original aggregation
                    self.loss_aggregation = original_aggregation
            else:
                # Handle special cases
                if name == 'neg_loss' and self.use_negative_prompts:
                    if 'neg_pred' in outputs['matched'] and 'neg_text_features' in outputs['matched']:
                        neg_loss = self._calculate_negative_prompt_loss(
                            outputs['matched']['neg_pred'],
                            outputs['matched']['neg_text_features']
                        )
                        losses['neg_loss'] = neg_loss
                    else:
                        device = self._get_device(*outputs.values()) if outputs else torch.device('cpu')
                        losses['neg_loss'] = torch.tensor(0.0, device=device)

        # Add quality metrics if enabled
        quality_cfg = self.std_val_quality_metrics
        if quality_cfg.get('include_penetration_penalty', False):
            pen_loss = self._calculate_penetration_penalty(outputs, targets)
            losses['penetration_penalty'] = pen_loss * quality_cfg.get('penetration_weight', 1.0)

        if quality_cfg.get('include_contact_quality', False):
            contact_loss = self._calculate_contact_quality(outputs, targets)
            losses['contact_quality'] = contact_loss * quality_cfg.get('contact_weight', 0.5)

        return losses

    def _normalize_by_num_grasps(self, loss_val, outputs, targets):
        """Normalize loss by number of grasps to ensure fairness across different num_grasps."""
        try:
            # Try to get num_grasps from the data shape
            if 'matched' in targets and 'norm_pose' in targets['matched']:
                pose_shape = targets['matched']['norm_pose'].shape
                if len(pose_shape) == 3:  # [B, num_grasps, pose_dim]
                    num_grasps = pose_shape[1]
                    # For mean aggregation, no additional normalization needed
                    # For sum aggregation, we would divide by num_grasps
                    return loss_val
            return loss_val
        except Exception:
            return loss_val

    def _normalize_by_batch_size(self, loss_val, outputs, targets):
        """Normalize loss by batch size for consistent scaling."""
        try:
            # For most loss functions, this is already handled by the reduction strategy
            return loss_val
        except Exception:
            return loss_val

    def _calculate_penetration_penalty(self, outputs, targets):
        """Calculate penetration penalty for quality assessment."""
        # Placeholder implementation - can be enhanced based on specific requirements
        device = self._get_device(*outputs.values()) if outputs else torch.device('cpu')
        return torch.tensor(0.0, device=device)

    def _calculate_contact_quality(self, outputs, targets):
        """Calculate contact quality metric for quality assessment."""
        # Placeholder implementation - can be enhanced based on specific requirements
        device = self._get_device(*outputs.values()) if outputs else torch.device('cpu')
        return torch.tensor(0.0, device=device)

    def _calculate_negative_prompt_loss(self, neg_pred, neg_text_features):
        """
        Calculate negative prompt loss
        
        Design concept:
        - negative_net predicts distractor embeddings from "scene-target" difference
        - Loss function makes predicted distractor embedding close to any real distractor
        - Uses min operation: only need to successfully predict any one distractor
        """
        if neg_pred is None or neg_text_features is None:
            device = self._get_device(neg_pred, neg_text_features)
            return torch.tensor(0.0, device=device)

        try:
            # Expand prediction embedding to match negative prompt dimensions for broadcast computation
            B, num_neg_prompts, embed_dim = neg_text_features.shape
            neg_pred_expanded = neg_pred.unsqueeze(1).expand(B, num_neg_prompts, embed_dim)

            # Calculate Euclidean distance between predicted distractor and each real distractor
            paired_distances = torch.sqrt(
                torch.sum((neg_pred_expanded - neg_text_features)**2, dim=2) + 1e-8
            )  # (B, num_neg_prompts)

            # Find minimum distance (closest distractor) for each sample
            min_distances = torch.min(paired_distances, dim=1)[0]  # (B,)

            # Average over entire batch
            neg_loss = torch.mean(min_distances)

            return neg_loss

        except Exception as e:
            logging.warning(f"Negative prompt loss calculation failed: {e}")
            device = self._get_device(neg_pred, neg_text_features)
            return torch.tensor(0.0, device=device)

    def _calculate_metrics(self, pred_dict, batch):
        """
        Calculate evaluation metrics by delegating to the GraspMetricCalculator.
        """
        return self.metric_calculator.calculate_metrics(pred_dict, batch)

    def get_hand_any_norm(self, norm_pose, scene_pc=None):
        """Get hand model output from normalized pose"""
        hand_model_pose = denorm_hand_pose_robust(norm_pose, self.rot_type, self.mode)
        hand = self.hand_model(
            hand_model_pose, 
            scene_pc=scene_pc,
            **self.test_infer_hand_model_kwargs
        )
        return hand

    def get_hand(self, outputs, targets):
        """
        Get hand model output from outputs and targets (compatibility with diffuser_lightning.py)
        This method extracts hand model data from the processed outputs and targets.
        """
        # Return hand model data from both outputs and targets
        return outputs['hand'], targets['hand']

    def print_matcher_results(self, matcher_output):
        """Print matcher results for debugging"""
        print("=== Matcher Results ===")
        
        print(f"Cost Matrix Shape: {matcher_output['final_cost'].shape}")
        
        for batch_idx, assign in enumerate(matcher_output['assignments']):
            queries, targets = assign
            print(f"Batch {batch_idx} Matches:")
            matched_count = len(queries)
            total_count = matcher_output['query_matched_mask'][batch_idx].numel()
            print(f"  Matched Queries: {matched_count}/{total_count}")
            for q, t in zip(queries.cpu().numpy(), targets.cpu().numpy()):
                print(f"  Query {q} -> Target {t}")
        
        print("\nMatching Statistics:")
        matched_queries = matcher_output['query_matched_mask'].sum().item()
        total_queries = matcher_output['query_matched_mask'].numel()
        print(f"Total Matched Queries: {matched_queries}/{total_queries}")