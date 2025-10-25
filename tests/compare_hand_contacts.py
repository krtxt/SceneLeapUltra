#!/usr/bin/env python

"""对比当前 HandModel 与备份版本在多抓取接触点上的一致性。"""

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
# 添加项目根目录到路径，避免相对导入失败
sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError as exc:
    raise SystemExit("需要安装 torch 才能运行该脚本。") from exc

try:
    from pytorch3d.transforms import matrix_to_quaternion
except ImportError as exc:
    raise SystemExit("需要安装 pytorch3d 才能运行该脚本。") from exc

from utils.hand_model import HandModel
from utils.hand_types import HandModelType


def _load_backup_hand_model():
    backup_path = ROOT / "experiments/dit_no_mask_with_text/backups/v1/utils/hand_model.py"
    if not backup_path.exists():
        raise FileNotFoundError(f"未找到备份 HandModel 文件: {backup_path}")

    spec = importlib.util.spec_from_file_location("hand_model_backup", backup_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    if not hasattr(module, "HandModel"):
        raise AttributeError(f"{backup_path} 中未定义 HandModel 类。")

    return module.HandModel


def _build_axis_pose(model: HandModel, batch_size: int, num_grasps: int) -> torch.Tensor:
    pose_dim = 3 + model.n_dofs + 3  # 平移 + 关节角 + 轴角旋转
    pose = torch.zeros(batch_size, num_grasps, pose_dim, device=model.device)
    joint_angles = model.default_joint_angles.view(1, 1, -1).expand(batch_size, num_grasps, -1)
    pose[..., 3 : 3 + model.n_dofs] = joint_angles
    return pose


def main() -> None:
    parser = argparse.ArgumentParser(description="比较新旧 HandModel 接触点计算结果。")
    parser.add_argument("--batch-size", type=int, default=2, help="批次大小（物体数量）")
    parser.add_argument("--num-grasps", type=int, default=3, help="每个物体的抓取数量")
    parser.add_argument(
        "--contacts-per-finger", type=int, default=2, help="每根手指采样的接触点数量"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="可选：将比较结果保存为 .pt 文件",
    )
    args = parser.parse_args()

    total_batch = args.batch_size * args.num_grasps

    new_model = HandModel(
        hand_model_type=HandModelType.LEAP,
        rot_type="axis",
        device="cpu",
    )

    BackupHandModel = _load_backup_hand_model()
    backup_model = BackupHandModel(
        hand_model_type=HandModelType.LEAP,
        device="cpu",
    )

    pose_axis = _build_axis_pose(new_model, args.batch_size, args.num_grasps)
    contact_indices = new_model.sample_contact_points(
        total_batch, n_contacts_per_finger=args.contacts_per_finger
    )

    new_model.set_parameters(pose_axis, contact_indices)

    # 构造旧模型输入（四元数旋转）
    pose_axis_flat = pose_axis.view(total_batch, -1)
    translations = pose_axis_flat[:, :3]
    joint_angles = pose_axis_flat[:, 3 : 3 + new_model.n_dofs]
    global_rot = new_model.global_rotation.view(total_batch, 3, 3)
    quaternions = matrix_to_quaternion(global_rot)

    pose_quat_flat = torch.cat([translations, quaternions, joint_angles], dim=-1)
    pose_quat = pose_quat_flat.view(
        args.batch_size, args.num_grasps, -1
    )  # 形状: (B, num_grasps, 3 + 4 + n_dofs)

    backup_model.set_parameters(pose_quat, contact_indices)

    contacts_new = new_model.contact_points
    contacts_backup = backup_model.contact_points

    if contacts_new is None or contacts_backup is None:
        raise RuntimeError("某个模型未生成接触点结果。")

    diff = torch.abs(contacts_new - contacts_backup)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"total_batch={total_batch}, num_contacts={contacts_new.shape[1]}")
    print(f"max_abs_diff={max_diff:.6e}")
    print(f"mean_abs_diff={mean_diff:.6e}")

    if args.out is not None:
        payload = {
            "contacts_new": contacts_new.cpu(),
            "contacts_backup": contacts_backup.cpu(),
            "diff": diff.cpu(),
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.out)
        print(f"结果已保存到 {args.out}")


if __name__ == "__main__":
    main()
