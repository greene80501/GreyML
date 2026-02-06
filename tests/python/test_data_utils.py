import numpy as np

from greyml.tensor import Tensor
from greyml.data.dataset import Dataset
from greyml.data.dataloader import DataLoader


class DummyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return Tensor(np.array([idx], dtype=np.float32))


def test_dataloader_batches_and_len():
    loader = DataLoader(DummyDataset(), batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(loader) == 2
    assert len(batches) == 2
    np.testing.assert_array_equal(batches[0].numpy().flatten(), np.array([0, 1], dtype=np.float32))
    np.testing.assert_array_equal(batches[1].numpy().flatten(), np.array([2, 3], dtype=np.float32))


def test_collate_tuples():
    data = [(Tensor(np.array([1], dtype=np.float32)), Tensor(np.array([2], dtype=np.float32))) for _ in range(2)]
    loader = DataLoader(data, batch_size=2, shuffle=False)
    x_batch, y_batch = next(iter(loader))
    np.testing.assert_array_equal(x_batch.numpy().flatten(), np.array([1, 1], dtype=np.float32))
    np.testing.assert_array_equal(y_batch.numpy().flatten(), np.array([2, 2], dtype=np.float32))
