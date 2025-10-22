import torch

from models.decoder.dit_memory_optimization import BatchProcessor


def test_batch_processor_preserves_original_order():
    processor = BatchProcessor(
        max_batch_size=3,
        max_sequence_length=16,
        default_d_model=64,
        default_num_layers=2,
    )
    inputs = [
        torch.zeros(1, 5, 3),
        torch.zeros(1, 2, 3),
        torch.zeros(1, 7, 3),
    ]

    call_sizes = []

    def forward_fn(batch):
        call_sizes.append(len(batch))
        return [torch.tensor(tensor.shape[1]) for tensor in batch]

    outputs = processor.process_variable_length_batch(forward_fn, inputs, max_memory_gb=10.0)
    recovered_lengths = [int(value.item()) for value in outputs]

    assert recovered_lengths == [5, 2, 7]
    assert sum(call_sizes) >= len(inputs)


def test_batch_processor_honors_memory_limit():
    processor = BatchProcessor(max_batch_size=4, max_sequence_length=16)
    inputs = [
        torch.zeros(1, 5, 3),
        torch.zeros(1, 6, 3),
        torch.zeros(1, 4, 3),
    ]

    call_sizes = []

    def forward_fn(batch):
        call_sizes.append(len(batch))
        return [torch.zeros(1) for _ in batch]

    processor.process_variable_length_batch(forward_fn, inputs, max_memory_gb=0.0005)

    assert all(size == 1 for size in call_sizes)
