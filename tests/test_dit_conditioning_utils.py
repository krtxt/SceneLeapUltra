import numpy as np
import pytest
import torch

from models.decoder.dit import _convert_to_tensor, _normalize_object_mask
from models.decoder.dit_validation import DiTConditioningError


def test_convert_to_tensor_from_list_of_tensors():
    device = torch.device("cpu")
    tensors = [torch.ones(3, 6), torch.zeros(3, 6)]
    result = _convert_to_tensor(tensors, device, torch.float32, "scene_pc")
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 3, 6)
    assert result.device == device


def test_convert_to_tensor_from_variable_length_tensors():
    device = torch.device("cpu")
    tensors = [torch.ones(5, 6), torch.zeros(2, 6), torch.ones(0, 6)]
    result = _convert_to_tensor(tensors, device, torch.float32, "scene_pc")
    assert result.shape == (3, 5, 6)
    assert torch.allclose(result[0, :5], torch.ones(5, 6))
    assert torch.all(result[1, 2:] == 0)
    assert torch.all(result[2] == 0)


def test_convert_to_tensor_from_numpy_array():
    device = torch.device("cpu")
    array = np.zeros((2, 4, 6), dtype=np.float32)
    result = _convert_to_tensor(array, device, torch.float32, "scene_pc")
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 4, 6)


def test_convert_to_tensor_invalid_type():
    device = torch.device("cpu")
    with pytest.raises(DiTConditioningError):
        _convert_to_tensor("invalid", device, torch.float32, "scene_pc")


def test_convert_to_tensor_mismatched_features_raises():
    device = torch.device("cpu")
    tensors = [torch.ones(5, 6), torch.zeros(5, 5)]
    with pytest.raises(DiTConditioningError):
        _convert_to_tensor(tensors, device, torch.float32, "scene_pc")


def test_normalize_object_mask_basic():
    scene_points = torch.zeros(2, 4, 3)
    mask = torch.tensor([[1, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.float32)
    normalized = _normalize_object_mask(scene_points, mask)
    assert normalized.shape == (2, 4, 1)
    assert torch.all(normalized[0, :, 0] == torch.tensor([1, 0, 1, 1], dtype=torch.float32))


def test_normalize_object_mask_rejects_empty():
    scene_points = torch.zeros(2, 4, 3)
    mask = torch.zeros(0)
    with pytest.raises(DiTConditioningError):
        _normalize_object_mask(scene_points, mask)


def test_normalize_object_mask_rejects_all_zero():
    scene_points = torch.zeros(2, 4, 3)
    mask = torch.zeros(2, 4)
    with pytest.raises(DiTConditioningError):
        _normalize_object_mask(scene_points, mask)


def test_normalize_object_mask_rejects_mismatch():
    scene_points = torch.zeros(2, 4, 3)
    mask = torch.ones(2, 5)
    with pytest.raises(DiTConditioningError):
        _normalize_object_mask(scene_points, mask)
