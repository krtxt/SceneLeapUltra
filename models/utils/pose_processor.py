from typing import Dict, Tuple
import torch

from utils.hand_model import HandModel
from utils.hand_helper import denorm_hand_pose_robust

# Define slice constants for pose components
TRANSLATION_SLICE = slice(0, 3)
QPOS_SLICE = slice(3, 19)
ROTATION_SLICE = slice(19, None)

class PoseProcessor:
    def __init__(self, hand_model: HandModel, rot_type: str, mode: str):
        self.hand_model = hand_model
        self.rot_type = rot_type
        self.mode = mode

    def process_train(self, pred_dict, batch, hand_model_kwargs):
        outputs, targets = self.get_hand_model_pose(pred_dict, batch)
        
        # Propagate negative prompt info from the original pred_dict,
        # as get_hand_model_pose restructures the dictionary.
        if 'neg_pred' in pred_dict and 'neg_text_features' in pred_dict:
            outputs['matched']['neg_pred'] = pred_dict['neg_pred']
            outputs['matched']['neg_text_features'] = pred_dict['neg_text_features']

        return self._prepare_hand_data(outputs, targets, hand_model_kwargs)

    def process_val(self, outputs, batch, matcher, hand_model_kwargs):
        outputs = self.get_hand_model_pose_test(outputs)
        assignments = matcher(outputs, batch)
        matched_preds, matched_targets = self.get_matched_by_assignment(outputs, batch, assignments)
        
        outputs['matched'] = matched_preds
        batch['matched'] = matched_targets
        batch['matched']['scene_pc'] = batch['scene_pc']
        
        # Propagate negative prompt info from model outputs if it exists
        if 'neg_pred' in outputs and 'neg_text_features' in outputs:
            outputs['matched']['neg_pred'] = outputs['neg_pred']
            outputs['matched']['neg_text_features'] = outputs['neg_text_features']
        
        return self._prepare_hand_data(outputs, batch, hand_model_kwargs)

    def process_test(self, outputs, batch, matcher):
        outputs = self.get_hand_model_pose_test(outputs)
        assignments = matcher(outputs, batch)
        matched_preds, matched_targets = self.get_matched_by_assignment(outputs, batch, assignments)
        
        outputs['matched'] = matched_preds
        batch['matched'] = matched_targets
        outputs['matched']['scene_pc'] = batch['scene_pc']
        outputs['matched']['obj_verts'] = batch['obj_verts']
        outputs['matched']['obj_faces'] = batch['obj_faces']
        
        return outputs, batch

    def process_infer(self, outputs, targets, matcher, hand_model_kwargs):
        outputs = self.get_hand_model_pose_test(outputs)
        assignments = matcher(outputs, targets)
        self.print_matcher_results(assignments)
        matched_preds, matched_targets = self.get_matched_by_assignment(outputs, targets, assignments)
        outputs['matched'] = matched_preds
        targets['matched'] = matched_targets
        targets['matched']['scene_pc'] = targets['scene_pc']
        if 'neg_pred' in outputs and 'neg_text_features' in outputs:
            outputs['matched']['neg_pred'] = outputs['neg_pred']
            outputs['matched']['neg_text_features'] = outputs['neg_text_features']
        outputs, targets = self._prepare_hand_data(outputs, targets, hand_model_kwargs)
        return matched_preds, matched_targets, outputs, targets

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

    def get_hand_model_pose(self, outputs, targets):
        targets['translation_norm'] = targets['norm_pose'][..., TRANSLATION_SLICE]
        targets['qpos_norm'] = targets['norm_pose'][..., QPOS_SLICE]
        targets['rotation'] = targets['norm_pose'][..., ROTATION_SLICE]
        
        outputs['translation_norm'] = outputs['pred_pose_norm'][..., TRANSLATION_SLICE]
        outputs['qpos_norm'] = outputs['pred_pose_norm'][..., QPOS_SLICE]
        outputs['rotation'] = outputs['pred_pose_norm'][..., ROTATION_SLICE]
        
        hand_model_pose = denorm_hand_pose_robust(outputs["pred_pose_norm"], self.rot_type, self.mode)
        outputs['hand_model_pose'] = hand_model_pose
        outputs = {'matched': outputs}
        targets = {'matched': targets}
        return outputs, targets

    def get_hand_model_pose_test(self, outputs):
        if outputs['pred_pose_norm'].dim() < 3:
            outputs['pred_pose_norm'] = outputs['pred_pose_norm'].unsqueeze(1)
        
        outputs['translation_norm'] = outputs['pred_pose_norm'][..., TRANSLATION_SLICE]
        outputs['qpos_norm'] = outputs['pred_pose_norm'][..., QPOS_SLICE]
        outputs['rotation'] = outputs['pred_pose_norm'][..., ROTATION_SLICE]
       
        hand_model_pose = denorm_hand_pose_robust(outputs['pred_pose_norm'], self.rot_type, self.mode)
        outputs['hand_model_pose'] = hand_model_pose
        return outputs

    def _prepare_hand_data(self, outputs, targets, hand_model_kwargs: Dict[str, bool]):
        scene_pc_for_targets = targets.get('scene_pc') 
        if 'matched' in targets and 'scene_pc' in targets['matched']:
            scene_pc_for_targets = targets['matched']['scene_pc']

        scene_pc_for_outputs = scene_pc_for_targets
            
        targets['hand'] = self.hand_model(
            targets['matched']['hand_model_pose'], 
            scene_pc=scene_pc_for_targets,
            **hand_model_kwargs
        )
        outputs['hand'] = self.hand_model(
            outputs['matched']['hand_model_pose'], 
            scene_pc=scene_pc_for_outputs,
            **hand_model_kwargs
        )

        outputs['rot_type'] = self.rot_type
        return outputs, targets

    def get_matched_by_assignment(self, predictions: Dict, targets: Dict, assignment: Dict) -> Tuple[Dict, Dict]:
        per_query_gt_inds = assignment["per_query_gt_inds"]

        matched_preds, matched_targets = {}, {}

        pred_target_match_key_map = {
            "pred_pose_norm": "norm_pose",
            "hand_model_pose": "hand_model_pose",
        }

        for pred_key, target_key in pred_target_match_key_map.items():
            if pred_key not in predictions:
                continue

            pred = predictions[pred_key]
            target = targets[target_key]

            if pred.dim() == 2:
                pred = pred.unsqueeze(1)

            matched_preds[pred_key] = pred

            B, N_queries, _ = pred.shape
            _B, _N_targets, D_target = target.shape

            idx = per_query_gt_inds.unsqueeze(-1).expand(B, N_queries, D_target)

            gathered_targets = torch.gather(target, 1, idx)
            matched_targets[target_key] = gathered_targets

        return matched_preds, matched_targets 