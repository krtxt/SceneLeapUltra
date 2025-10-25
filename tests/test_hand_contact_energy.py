import pytest

from utils.hand_model import HandModel
from utils.hand_types import HandModelType


def _require_deps():
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch3d")
    return torch


def _build_multi_grasp_pose(model: HandModel, batch_size: int, num_grasps: int) -> torch.Tensor:
    torch = _require_deps()
    pose_dim = 3 + model.n_dofs + 3  # translation + joints + axis-angle
    hand_pose = torch.zeros(batch_size, num_grasps, pose_dim, dtype=torch.float, device=model.device)

    # Translation stays at the origin
    joint_angles = model.default_joint_angles.view(1, 1, -1).expand(batch_size, num_grasps, -1)

    hand_pose[..., 3 : 3 + model.n_dofs] = joint_angles
    # Axis-angle zero => identity rotation
    return hand_pose


@pytest.mark.parametrize("batch_size,num_grasps", [(1, 1), (2, 3)])
def test_contact_energy_shapes_and_finiteness(batch_size: int, num_grasps: int):
    torch = _require_deps()
    model = HandModel(
        hand_model_type=HandModelType.LEAP,
        rot_type="axis",
        device="cpu",
    )

    pose = _build_multi_grasp_pose(model, batch_size, num_grasps)
    total_batch = batch_size * num_grasps
    contact_indices = model.sample_contact_points(total_batch, n_contacts_per_finger=2)

    model.set_parameters(pose, contact_indices)

    finger_energy = model.cal_finger_finger_distance_energy()
    palm_energy = model.cal_finger_palm_distance_energy()

    assert finger_energy.shape == (total_batch,)
    assert palm_energy.shape == (total_batch,)

    assert torch.isfinite(finger_energy).all()
    assert torch.isfinite(palm_energy).all()
