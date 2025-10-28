import torch

from models.loss.set_ops import chamfer_set_loss, pairwise_cost
from models.metrics.set_metrics import SetMetricsEvaluator


def test_pairwise_cost_translation_only():
    prediction = {
        'translation_norm': torch.tensor([
            [[0.0, 0.0, 0.0],
             [1.0, 0.0, 0.0]]
        ], dtype=torch.float32)
    }
    targets = {
        'translation_norm': torch.tensor([
            [[0.0, 0.0, 0.0],
             [2.0, 0.0, 0.0]]
        ], dtype=torch.float32)
    }
    weights = {'translation': 1.0, 'rotation': 0.0, 'qpos': 0.0}

    cost = pairwise_cost(prediction, targets, weights=weights, rot_type='r6d')

    expected = torch.tensor([[[0.0, 2.0],
                              [1.0, 1.0]]], dtype=torch.float32)
    assert torch.allclose(cost, expected)


def test_chamfer_set_loss_with_mask():
    cost_matrix = torch.tensor([[[0.0, 2.0],
                                 [3.0, 4.0]]], dtype=torch.float32)
    valid_mask = torch.tensor([[True, False]])

    loss = chamfer_set_loss(cost_matrix, valid_mask=valid_mask)
    assert torch.isclose(loss, torch.tensor(1.5))


def test_set_metrics_evaluator_basic():
    translation = torch.tensor([
        [[0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0]]
    ], dtype=torch.float32)
    qpos = torch.zeros(1, 2, 16, dtype=torch.float32)
    rotation = torch.zeros(1, 2, 6, dtype=torch.float32)
    pred_pose = torch.cat([translation, qpos, rotation], dim=-1)

    pred_dict = {
        'pred_pose_norm': pred_pose,
        'translation_norm': translation,
        'qpos_norm': qpos,
        'rotation': rotation,
    }
    batch = {
        'norm_pose': pred_pose.clone(),
    }

    evaluator = SetMetricsEvaluator(
        {
            'enable': True,
            'cov_threshold': 0.5,
            'max_samples': 0,
            'log': {'cov': True, 'mmd': True, 'nnd': True},
        },
        rot_type='r6d',
        cost_weights={'translation': 1.0, 'rotation': 0.0, 'qpos': 0.0},
    )

    metrics = evaluator.compute(pred_dict, batch)

    assert metrics
    assert torch.isclose(torch.tensor(metrics['cov@0.50']), torch.tensor(1.0))
    assert torch.isclose(torch.tensor(metrics['mmd']), torch.tensor(0.0))
    assert torch.isclose(torch.tensor(metrics['nnd_mean']), torch.tensor(1.0))
    assert torch.isclose(torch.tensor(metrics['nnd_std']), torch.tensor(0.0))
    assert torch.isclose(torch.tensor(metrics['nnd_cv']), torch.tensor(0.0))
