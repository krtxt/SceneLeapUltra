"""
DiT Model Input Validation and Error Handling

This module provides comprehensive validation and error handling for the DiT model,
including input validation, device consistency checks, and graceful fallback mechanisms.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch


# Custom Exception Classes for DiT-specific errors
class DiTValidationError(Exception):
    """Base exception for DiT model validation errors."""
    pass


class DiTInputError(DiTValidationError):
    """Exception raised for invalid input tensors."""
    pass


class DiTDimensionError(DiTValidationError):
    """Exception raised for tensor dimension mismatches."""
    pass


class DiTDeviceError(DiTValidationError):
    """Exception raised for device consistency issues."""
    pass


class DiTConditioningError(DiTValidationError):
    """Exception raised for conditioning-related errors."""
    pass


class DiTConfigurationError(DiTValidationError):
    """Exception raised for configuration-related errors."""
    pass


@dataclass
class ValidationResult:
    """Result of input validation with details about any issues found."""
    is_valid: bool
    errors: list
    warnings: list
    corrected_inputs: Optional[Dict] = None


class DiTInputValidator:
    """
    Comprehensive input validator for DiT model.
    
    Validates tensor dimensions, types, device consistency, and provides
    graceful fallback mechanisms for various error conditions.
    """
    
    def __init__(self, d_x: int, rot_type: str, max_sequence_length: int = 1000):
        """
        Initialize the validator.
        
        Args:
            d_x: Expected pose dimension (23 for quat, 25 for r6d)
            rot_type: Rotation representation type ('quat' or 'r6d')
            max_sequence_length: Maximum allowed sequence length
        """
        self.d_x = d_x
        self.rot_type = rot_type
        self.max_sequence_length = max_sequence_length
        self.logger = logging.getLogger(__name__)
        
        # Valid rotation types and their dimensions
        self.rot_to_dim = {'quat': 23, 'r6d': 25}
    
    def validate_input_tensor(self, x_t: torch.Tensor, 
                            allow_auto_correction: bool = True) -> ValidationResult:
        """
        Validate the main input tensor x_t.
        
        Args:
            x_t: Input grasp poses tensor
            allow_auto_correction: Whether to attempt automatic corrections
            
        Returns:
            ValidationResult with validation status and any corrections
        """
        errors = []
        warnings = []
        corrected_inputs = {}
        
        # Check if tensor exists and is valid
        if x_t is None:
            errors.append("Input tensor x_t is None")
            return ValidationResult(False, errors, warnings)
        
        if not isinstance(x_t, torch.Tensor):
            errors.append(f"Input x_t must be torch.Tensor, got {type(x_t)}")
            return ValidationResult(False, errors, warnings)
        
        # Check tensor dimensions
        if x_t.dim() not in [2, 3]:
            errors.append(f"Input tensor must be 2D or 3D, got {x_t.dim()}D with shape {x_t.shape}")
            return ValidationResult(False, errors, warnings)
        
        # Validate tensor shape based on dimensions
        if x_t.dim() == 2:
            # Single grasp format: (B, d_pose)
            batch_size, pose_dim = x_t.shape
            if pose_dim != self.d_x:
                error_msg = (f"Single grasp pose dimension {pose_dim} doesn't match expected {self.d_x} "
                           f"for rot_type '{self.rot_type}'")
                errors.append(error_msg)
        
        elif x_t.dim() == 3:
            # Multi-grasp format: (B, num_grasps, d_pose)
            batch_size, num_grasps, pose_dim = x_t.shape
            
            if pose_dim != self.d_x:
                error_msg = (f"Multi-grasp pose dimension {pose_dim} doesn't match expected {self.d_x} "
                           f"for rot_type '{self.rot_type}'")
                errors.append(error_msg)
            
            if num_grasps > self.max_sequence_length:
                if allow_auto_correction:
                    warnings.append(f"Sequence length {num_grasps} exceeds maximum {self.max_sequence_length}, "
                                  f"truncating to {self.max_sequence_length}")
                    corrected_inputs['x_t'] = x_t[:, :self.max_sequence_length, :]
                else:
                    errors.append(f"Sequence length {num_grasps} exceeds maximum {self.max_sequence_length}")
            
            if num_grasps == 0:
                errors.append("Number of grasps cannot be zero")
        
        # Check for invalid values
        if torch.isnan(x_t).any():
            if allow_auto_correction:
                warnings.append("Input tensor contains NaN values, replacing with zeros")
                corrected_x_t = x_t.clone()
                corrected_x_t[torch.isnan(corrected_x_t)] = 0.0
                corrected_inputs['x_t'] = corrected_x_t
            else:
                errors.append("Input tensor contains NaN values")
        
        if torch.isinf(x_t).any():
            if allow_auto_correction:
                warnings.append("Input tensor contains infinite values, clamping to finite range")
                corrected_x_t = x_t.clone()
                corrected_x_t = torch.clamp(corrected_x_t, -1e6, 1e6)
                corrected_inputs['x_t'] = corrected_x_t
            else:
                errors.append("Input tensor contains infinite values")
        
        # Check tensor dtype
        if not x_t.dtype.is_floating_point:
            if allow_auto_correction:
                warnings.append(f"Converting input tensor from {x_t.dtype} to float32")
                corrected_inputs['x_t'] = x_t.float()
            else:
                errors.append(f"Input tensor must be floating point, got {x_t.dtype}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, corrected_inputs if corrected_inputs else None)
    
    def validate_timesteps(self, ts: torch.Tensor, batch_size: int,
                          allow_auto_correction: bool = True) -> ValidationResult:
        """
        Validate timestep tensor.
        
        Args:
            ts: Timestep tensor
            batch_size: Expected batch size
            allow_auto_correction: Whether to attempt automatic corrections
            
        Returns:
            ValidationResult with validation status and any corrections
        """
        errors = []
        warnings = []
        corrected_inputs = {}
        
        if ts is None:
            errors.append("Timestep tensor ts is None")
            return ValidationResult(False, errors, warnings)
        
        if not isinstance(ts, torch.Tensor):
            errors.append(f"Timesteps ts must be torch.Tensor, got {type(ts)}")
            return ValidationResult(False, errors, warnings)
        
        # Check dimensions
        if ts.dim() != 1:
            if allow_auto_correction and ts.dim() == 0:
                warnings.append("Converting scalar timestep to 1D tensor")
                corrected_inputs['ts'] = ts.unsqueeze(0).expand(batch_size)
            else:
                errors.append(f"Timesteps must be 1D tensor, got {ts.dim()}D with shape {ts.shape}")
        
        # Check batch size consistency
        elif ts.shape[0] != batch_size:
            if allow_auto_correction and ts.shape[0] == 1:
                warnings.append(f"Broadcasting timestep from size 1 to batch size {batch_size}")
                corrected_inputs['ts'] = ts.expand(batch_size)
            else:
                errors.append(f"Timestep batch size {ts.shape[0]} doesn't match input batch size {batch_size}")
        
        # Check for valid timestep values
        if ts.numel() > 0:
            if torch.isnan(ts).any():
                errors.append("Timestep tensor contains NaN values")
            
            if torch.isinf(ts).any():
                errors.append("Timestep tensor contains infinite values")
            
            if (ts < 0).any():
                if allow_auto_correction:
                    warnings.append("Negative timestep values found, clamping to 0")
                    corrected_ts = torch.clamp(ts, min=0)
                    corrected_inputs['ts'] = corrected_ts
                else:
                    errors.append("Timestep values must be non-negative")
        
        # Check tensor dtype
        if not ts.dtype.is_floating_point and ts.dtype != torch.long:
            if allow_auto_correction:
                warnings.append(f"Converting timestep tensor from {ts.dtype} to long")
                corrected_inputs['ts'] = ts.long()
            else:
                errors.append(f"Timestep tensor must be floating point or long, got {ts.dtype}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, corrected_inputs if corrected_inputs else None)
    
    def validate_conditioning_data(self, data: Dict, 
                                 allow_auto_correction: bool = True) -> ValidationResult:
        """
        Validate conditioning data dictionary.
        
        Args:
            data: Conditioning data dictionary
            allow_auto_correction: Whether to attempt automatic corrections
            
        Returns:
            ValidationResult with validation status and any corrections
        """
        errors = []
        warnings = []
        corrected_inputs = {}
        
        if data is None:
            errors.append("Conditioning data is None")
            return ValidationResult(False, errors, warnings)
        
        if not isinstance(data, dict):
            errors.append(f"Conditioning data must be dict, got {type(data)}")
            return ValidationResult(False, errors, warnings)
        
        # Validate scene conditioning
        if 'scene_cond' in data:
            scene_cond = data['scene_cond']
            if scene_cond is not None:
                if not isinstance(scene_cond, torch.Tensor):
                    errors.append(f"scene_cond must be torch.Tensor, got {type(scene_cond)}")
                elif scene_cond.dim() != 3:
                    errors.append(f"scene_cond must be 3D tensor (B, N_points, d_model), got {scene_cond.dim()}D")
                elif torch.isnan(scene_cond).any():
                    if allow_auto_correction:
                        warnings.append("scene_cond contains NaN values, replacing with zeros")
                        corrected_scene = scene_cond.clone()
                        corrected_scene[torch.isnan(corrected_scene)] = 0.0
                        if 'data' not in corrected_inputs:
                            corrected_inputs['data'] = data.copy()
                        corrected_inputs['data']['scene_cond'] = corrected_scene
                    else:
                        errors.append("scene_cond contains NaN values")
        
        # Validate text conditioning
        if 'text_cond' in data:
            text_cond = data['text_cond']
            if text_cond is not None:
                if not isinstance(text_cond, torch.Tensor):
                    errors.append(f"text_cond must be torch.Tensor, got {type(text_cond)}")
                elif text_cond.dim() not in [2, 3]:
                    errors.append(f"text_cond must be 2D or 3D tensor, got {text_cond.dim()}D")
                elif torch.isnan(text_cond).any():
                    if allow_auto_correction:
                        warnings.append("text_cond contains NaN values, setting to None")
                        if 'data' not in corrected_inputs:
                            corrected_inputs['data'] = data.copy()
                        corrected_inputs['data']['text_cond'] = None
                    else:
                        errors.append("text_cond contains NaN values")
        
        # Validate text mask
        if 'text_mask' in data:
            text_mask = data['text_mask']
            if text_mask is not None:
                if not isinstance(text_mask, torch.Tensor):
                    errors.append(f"text_mask must be torch.Tensor, got {type(text_mask)}")
                elif text_mask.dim() != 2:
                    errors.append(f"text_mask must be 2D tensor, got {text_mask.dim()}D")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, corrected_inputs if corrected_inputs else None)
    
    def check_device_consistency(self, *tensors: torch.Tensor, 
                               target_device: Optional[torch.device] = None,
                               allow_auto_correction: bool = True) -> ValidationResult:
        """
        Check device consistency across tensors and optionally move to target device.
        
        Args:
            *tensors: Variable number of tensors to check
            target_device: Target device to move tensors to (optional)
            allow_auto_correction: Whether to automatically move tensors to consistent device
            
        Returns:
            ValidationResult with validation status and any corrections
        """
        errors = []
        warnings = []
        corrected_inputs = {}
        
        # Filter out None tensors
        valid_tensors = [t for t in tensors if t is not None and isinstance(t, torch.Tensor)]
        
        if not valid_tensors:
            return ValidationResult(True, errors, warnings)
        
        # Determine reference device
        if target_device is not None:
            reference_device = target_device
        else:
            reference_device = valid_tensors[0].device
        
        # Check device consistency
        inconsistent_tensors = []
        for i, tensor in enumerate(valid_tensors):
            if tensor.device != reference_device:
                inconsistent_tensors.append((i, tensor.device))
        
        if inconsistent_tensors:
            if allow_auto_correction:
                warnings.append(f"Moving {len(inconsistent_tensors)} tensors to device {reference_device}")
                # Note: Actual tensor movement should be done by the caller
                # We just report that correction is needed
                corrected_inputs['device_correction_needed'] = True
                corrected_inputs['target_device'] = reference_device
            else:
                device_info = ", ".join([f"tensor_{i}: {device}" for i, device in inconsistent_tensors])
                errors.append(f"Device inconsistency detected. Reference: {reference_device}, "
                            f"Inconsistent: {device_info}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, corrected_inputs if corrected_inputs else None)
    
    def validate_all_inputs(self, x_t: torch.Tensor, ts: torch.Tensor, data: Dict,
                          model_device: Optional[torch.device] = None,
                          allow_auto_correction: bool = True) -> Tuple[ValidationResult, Dict]:
        """
        Comprehensive validation of all inputs.
        
        Args:
            x_t: Input grasp poses tensor
            ts: Timestep tensor
            data: Conditioning data dictionary
            model_device: Model's current device
            allow_auto_correction: Whether to attempt automatic corrections
            
        Returns:
            Tuple of (ValidationResult, corrected_inputs_dict)
        """
        all_errors = []
        all_warnings = []
        all_corrections = {}
        
        # Validate input tensor
        x_result = self.validate_input_tensor(x_t, allow_auto_correction)
        all_errors.extend(x_result.errors)
        all_warnings.extend(x_result.warnings)
        if x_result.corrected_inputs:
            all_corrections.update(x_result.corrected_inputs)
        
        if not x_result.is_valid:
            # If input tensor is invalid, return early
            return ValidationResult(False, all_errors, all_warnings, all_corrections), all_corrections
        
        # Get batch size for timestep validation
        corrected_x_t = all_corrections.get('x_t', x_t)
        batch_size = corrected_x_t.shape[0]
        
        # Validate timesteps
        ts_result = self.validate_timesteps(ts, batch_size, allow_auto_correction)
        all_errors.extend(ts_result.errors)
        all_warnings.extend(ts_result.warnings)
        if ts_result.corrected_inputs:
            all_corrections.update(ts_result.corrected_inputs)
        
        # Validate conditioning data
        data_result = self.validate_conditioning_data(data, allow_auto_correction)
        all_errors.extend(data_result.errors)
        all_warnings.extend(data_result.warnings)
        if data_result.corrected_inputs:
            all_corrections.update(data_result.corrected_inputs)
        
        # Check device consistency
        tensors_to_check = [corrected_x_t, all_corrections.get('ts', ts)]
        corrected_data = all_corrections.get('data', data)
        if corrected_data and 'scene_cond' in corrected_data and corrected_data['scene_cond'] is not None:
            tensors_to_check.append(corrected_data['scene_cond'])
        if corrected_data and 'text_cond' in corrected_data and corrected_data['text_cond'] is not None:
            tensors_to_check.append(corrected_data['text_cond'])
        
        device_result = self.check_device_consistency(*tensors_to_check, 
                                                    target_device=model_device,
                                                    allow_auto_correction=allow_auto_correction)
        all_errors.extend(device_result.errors)
        all_warnings.extend(device_result.warnings)
        if device_result.corrected_inputs:
            all_corrections.update(device_result.corrected_inputs)
        
        is_valid = len(all_errors) == 0
        final_result = ValidationResult(is_valid, all_errors, all_warnings, all_corrections if all_corrections else None)
        
        return final_result, all_corrections


class DiTGracefulFallback:
    """
    Provides graceful fallback mechanisms for various DiT model failures.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def handle_conditioning_failure(self, data: Dict, error: Exception) -> Dict:
        """
        Handle conditioning failures by providing fallback conditioning.
        
        Args:
            data: Original conditioning data
            error: The exception that occurred
            
        Returns:
            Fallback conditioning dictionary
        """
        self.logger.warning(f"Conditioning failed: {error}. Using fallback conditioning.")
        
        fallback_data = data.copy() if data else {}
        
        # Ensure we have at least scene conditioning
        if 'scene_cond' not in fallback_data or fallback_data['scene_cond'] is None:
            self.logger.warning("No scene conditioning available, creating dummy scene features")
            # Create minimal dummy scene features
            batch_size = 1  # Default batch size
            if 'scene_pc' in data:
                batch_size = data['scene_pc'].shape[0] if data['scene_pc'] is not None else 1
            
            # Create dummy scene features (B, N_points, d_model)
            dummy_scene = torch.zeros(batch_size, 1024, 512)  # Reasonable defaults
            fallback_data['scene_cond'] = dummy_scene
        
        # Remove problematic text conditioning
        fallback_data['text_cond'] = None
        fallback_data['text_mask'] = None
        if 'neg_pred' in fallback_data:
            fallback_data['neg_pred'] = None
        if 'neg_text_features' in fallback_data:
            fallback_data['neg_text_features'] = None
        
        return fallback_data
    
    def handle_device_mismatch(self, tensors: Dict[str, torch.Tensor], 
                             target_device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Handle device mismatches by moving tensors to target device.
        
        Args:
            tensors: Dictionary of tensor name -> tensor
            target_device: Target device
            
        Returns:
            Dictionary with tensors moved to target device
        """
        corrected_tensors = {}
        
        for name, tensor in tensors.items():
            if tensor is not None and isinstance(tensor, torch.Tensor):
                if tensor.device != target_device:
                    self.logger.debug(f"Moving {name} from {tensor.device} to {target_device}")
                    try:
                        corrected_tensors[name] = tensor.to(target_device)
                    except Exception as e:
                        self.logger.error(f"Failed to move {name} to {target_device}: {e}")
                        corrected_tensors[name] = tensor  # Keep original if move fails
                else:
                    corrected_tensors[name] = tensor
            else:
                corrected_tensors[name] = tensor
        
        return corrected_tensors
    
    def handle_dimension_mismatch(self, x_t: torch.Tensor, expected_dim: int) -> torch.Tensor:
        """
        Handle dimension mismatches in input tensors.
        
        Args:
            x_t: Input tensor with wrong dimensions
            expected_dim: Expected last dimension
            
        Returns:
            Corrected tensor or raises exception if cannot be corrected
        """
        if x_t.shape[-1] == expected_dim:
            return x_t
        
        self.logger.warning(f"Dimension mismatch: got {x_t.shape[-1]}, expected {expected_dim}")
        
        # Try to pad or truncate the last dimension
        current_dim = x_t.shape[-1]
        
        if current_dim < expected_dim:
            # Pad with zeros
            pad_size = expected_dim - current_dim
            if x_t.dim() == 2:
                padding = torch.zeros(x_t.shape[0], pad_size, device=x_t.device, dtype=x_t.dtype)
                corrected = torch.cat([x_t, padding], dim=1)
            elif x_t.dim() == 3:
                padding = torch.zeros(x_t.shape[0], x_t.shape[1], pad_size, device=x_t.device, dtype=x_t.dtype)
                corrected = torch.cat([x_t, padding], dim=2)
            else:
                raise DiTDimensionError(f"Cannot handle dimension correction for {x_t.dim()}D tensor")
            
            self.logger.warning(f"Padded tensor from {current_dim} to {expected_dim} dimensions")
            return corrected
        
        else:
            # Truncate
            if x_t.dim() == 2:
                corrected = x_t[:, :expected_dim]
            elif x_t.dim() == 3:
                corrected = x_t[:, :, :expected_dim]
            else:
                raise DiTDimensionError(f"Cannot handle dimension correction for {x_t.dim()}D tensor")
            
            self.logger.warning(f"Truncated tensor from {current_dim} to {expected_dim} dimensions")
            return corrected


def validate_dit_inputs(x_t: torch.Tensor, ts: torch.Tensor, data: Dict,
                       d_x: int, rot_type: str, model_device: Optional[torch.device] = None,
                       max_sequence_length: int = 1000,
                       allow_auto_correction: bool = True,
                       logger: Optional[logging.Logger] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Convenience function for comprehensive DiT input validation.
    
    Args:
        x_t: Input grasp poses tensor
        ts: Timestep tensor
        data: Conditioning data dictionary
        d_x: Expected pose dimension
        rot_type: Rotation representation type
        model_device: Model's current device
        max_sequence_length: Maximum allowed sequence length
        allow_auto_correction: Whether to attempt automatic corrections
        logger: Optional logger instance
        
    Returns:
        Tuple of (validated_x_t, validated_ts, validated_data)
        
    Raises:
        DiTValidationError: If validation fails and auto-correction is disabled
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    validator = DiTInputValidator(d_x, rot_type, max_sequence_length)
    fallback = DiTGracefulFallback(logger)
    
    try:
        # Perform comprehensive validation
        result, corrections = validator.validate_all_inputs(
            x_t, ts, data, model_device, allow_auto_correction
        )
        
        # Log warnings
        for warning in result.warnings:
            logger.warning(warning)
        
        # Handle errors
        if not result.is_valid:
            if allow_auto_correction:
                logger.error(f"Validation failed with errors: {result.errors}")
                raise DiTValidationError(f"Input validation failed: {'; '.join(result.errors)}")
            else:
                raise DiTValidationError(f"Input validation failed: {'; '.join(result.errors)}")
        
        # Apply corrections
        corrected_x_t = corrections.get('x_t', x_t)
        corrected_ts = corrections.get('ts', ts)
        corrected_data = corrections.get('data', data)
        
        # Handle device corrections
        if corrections.get('device_correction_needed', False):
            target_device = corrections['target_device']
            tensor_dict = {
                'x_t': corrected_x_t,
                'ts': corrected_ts
            }
            
            # Add conditioning tensors if they exist
            if corrected_data.get('scene_cond') is not None:
                tensor_dict['scene_cond'] = corrected_data['scene_cond']
            if corrected_data.get('text_cond') is not None:
                tensor_dict['text_cond'] = corrected_data['text_cond']
            
            corrected_tensors = fallback.handle_device_mismatch(tensor_dict, target_device)
            
            corrected_x_t = corrected_tensors['x_t']
            corrected_ts = corrected_tensors['ts']
            
            # Update data with corrected conditioning tensors
            if 'scene_cond' in corrected_tensors:
                corrected_data = corrected_data.copy()
                corrected_data['scene_cond'] = corrected_tensors['scene_cond']
            if 'text_cond' in corrected_tensors:
                corrected_data = corrected_data.copy()
                corrected_data['text_cond'] = corrected_tensors['text_cond']
        
        return corrected_x_t, corrected_ts, corrected_data
        
    except Exception as e:
        if isinstance(e, DiTValidationError):
            raise
        else:
            logger.error(f"Unexpected error during validation: {e}")
            raise DiTValidationError(f"Validation failed due to unexpected error: {e}")