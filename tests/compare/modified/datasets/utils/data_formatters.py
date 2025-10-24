"""
Data formatters for cached SceneLeapPro datasets.

This module provides data formatting utilities that handle the conversion
of cached data into the required output format for training and evaluation.
Maintains exact compatibility with original formatting logic.
"""

import torch
import logging
from typing import Dict, Any, List
from datasets.utils.hdf5_utils import get_default_error_values
from .constants import (
    STANDARD_CACHE_KEYS, FORMATCH_VAL_CACHE_KEYS, FORMATCH_TEST_CACHE_KEYS,
    DEFAULT_ERROR_PROMPT, DEFAULT_EMPTY_PROMPT, get_cache_keys_for_mode,
    get_default_error_values as get_constants_error_values
)


class DataFormatter:
    """
    Handles data formatting for cached datasets.

    This class provides static methods for formatting both normal and error data
    from cached datasets, ensuring identical output format to the original implementation.
    """

    @staticmethod
    def format_normal_data(cached_data: Dict[str, Any],
                          num_neg_prompts: int) -> Dict[str, Any]:
        """
        Format normal cached data - must preserve exact output.

        Args:
            cached_data: Raw cached data from HDF5 file
            num_neg_prompts: Number of negative prompts expected

        Returns:
            dict: Formatted data with required fields:
                - scene_pc: 6D point cloud data (xyz+rgb)
                - hand_model_pose: Hand pose data
                - se3: SE3 transformation matrix
                - positive_prompt: Positive prompt string
                - negative_prompts: List of negative prompt strings
        """
        default_values = get_default_error_values(num_neg_prompts)
        return {
            'scene_pc': cached_data.get('scene_pc', default_values['scene_pc']),
            'hand_model_pose': cached_data.get('hand_model_pose', default_values['hand_model_pose']),
            'se3': cached_data.get('se3', default_values['se3']),
            'positive_prompt': cached_data.get('positive_prompt', default_values['positive_prompt']),
            'negative_prompts': cached_data.get('negative_prompts', default_values['negative_prompts']),
        }

    @staticmethod
    def format_error_data(cached_data: Dict[str, Any],
                         num_neg_prompts: int) -> Dict[str, Any]:
        """
        Format error data - must preserve exact output.

        Args:
            cached_data: Raw cached data containing error information
            num_neg_prompts: Number of negative prompts expected

        Returns:
            dict: Formatted error data with fallback values and error field
        """
        default_values = get_default_error_values(num_neg_prompts)
        return {
            'scene_pc': cached_data.get('scene_pc', default_values['scene_pc']),
            'hand_model_pose': cached_data.get('hand_model_pose', default_values['hand_model_pose']),
            'se3': cached_data.get('se3', default_values['se3']),
            'positive_prompt': cached_data.get('positive_prompt', default_values['positive_prompt']),
            'negative_prompts': cached_data.get('negative_prompts', default_values['negative_prompts']),
            'error': cached_data['error']
        }


class ForMatchDataFormatter(DataFormatter):
    """
    Specialized formatter for ForMatch datasets.

    Extends the base DataFormatter to handle additional fields required
    for ForMatch training, including object mesh data and variable-length arrays.
    """

    @staticmethod
    def get_formatch_default_error_values(num_neg_prompts: int, cache_mode: str = "val") -> Dict[str, Any]:
        """
        Get default values for error cases in ForMatch dataset.

        Args:
            num_neg_prompts: Number of negative prompts
            cache_mode: Cache mode - "val" for validation or "test" for testing

        Returns:
            dict: Default error values based on cache_mode from constants
        """
        # Get base error values from constants
        base_values = get_constants_error_values(num_neg_prompts)

        # Additional fields for test mode
        if cache_mode == "test":
            base_values.update({
                'obj_code': 'unknown',
                'scene_id': 'unknown',
                'category_id_from_object_index': -1,
                'depth_view_index': -1,
            })

        return base_values

    @staticmethod
    def format_formatch_normal_data(cached_data: Dict[str, Any],
                                   num_neg_prompts: int,
                                   cache_mode: str = "val") -> Dict[str, Any]:
        """
        Format normal cached data for ForMatch dataset.

        Args:
            cached_data: Raw cached data from HDF5 file
            num_neg_prompts: Number of negative prompts expected
            cache_mode: Cache mode - "val" or "test"

        Returns:
            dict: Formatted data with all required ForMatch fields
        """
        default_values = ForMatchDataFormatter.get_formatch_default_error_values(num_neg_prompts, cache_mode)

        result = {
            'scene_pc': cached_data.get('scene_pc', default_values['scene_pc']),
            'hand_model_pose': cached_data.get('hand_model_pose', default_values['hand_model_pose']),
            'se3': cached_data.get('se3', default_values['se3']),
            'positive_prompt': cached_data.get('positive_prompt', default_values['positive_prompt']),
            'negative_prompts': cached_data.get('negative_prompts', default_values['negative_prompts']),
            'obj_verts': cached_data.get('obj_verts', default_values['obj_verts']),
            'obj_faces': cached_data.get('obj_faces', default_values['obj_faces']),
        }

        # Add test-specific fields if in test mode
        if cache_mode == "test":
            result.update({
                'obj_code': cached_data.get('obj_code', default_values['obj_code']),
                'scene_id': cached_data.get('scene_id', default_values['scene_id']),
                'category_id_from_object_index': cached_data.get('category_id_from_object_index', default_values['category_id_from_object_index']),
                'depth_view_index': cached_data.get('depth_view_index', default_values['depth_view_index']),
            })

        return result

    @staticmethod
    def format_formatch_error_data(cached_data: Dict[str, Any],
                                  num_neg_prompts: int,
                                  cache_mode: str = "val") -> Dict[str, Any]:
        """
        Format error data for ForMatch dataset.

        Args:
            cached_data: Raw cached data containing error information
            num_neg_prompts: Number of negative prompts expected
            cache_mode: Cache mode - "val" or "test"

        Returns:
            dict: Formatted error data with fallback values and error field
        """
        default_values = ForMatchDataFormatter.get_formatch_default_error_values(num_neg_prompts, cache_mode)

        result = {
            'scene_pc': cached_data.get('scene_pc', default_values['scene_pc']),
            'hand_model_pose': cached_data.get('hand_model_pose', default_values['hand_model_pose']),
            'se3': cached_data.get('se3', default_values['se3']),
            'positive_prompt': cached_data.get('positive_prompt', default_values['positive_prompt']),
            'negative_prompts': cached_data.get('negative_prompts', default_values['negative_prompts']),
            'obj_verts': cached_data.get('obj_verts', default_values['obj_verts']),
            'obj_faces': cached_data.get('obj_faces', default_values['obj_faces']),
            'error': cached_data['error']
        }

        # Add test-specific fields if in test mode
        if cache_mode == "test":
            result.update({
                'obj_code': cached_data.get('obj_code', default_values['obj_code']),
                'scene_id': cached_data.get('scene_id', default_values['scene_id']),
                'category_id_from_object_index': cached_data.get('category_id_from_object_index', default_values['category_id_from_object_index']),
                'depth_view_index': cached_data.get('depth_view_index', default_values['depth_view_index']),
            })

        return result

    @staticmethod
    def get_formatch_cache_keys(cache_mode: str = "val") -> List[str]:
        """
        Get list of keys to cache for ForMatch training.

        Args:
            cache_mode: Cache mode - "val" or "test"

        Returns:
            list: Cache keys for ForMatch dataset based on cache_mode from constants
        """
        return get_cache_keys_for_mode(cache_mode)