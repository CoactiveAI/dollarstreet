from typing import Optional

from torch.utils.data import Dataset, DataLoader

import constants as c


def get_loader(
    dataset: Dataset,
    batch_size: Optional[int] = c.BATCH_SIZE,
    shuffle: Optional[bool] = c.SHUFFLE,
    num_workers: Optional[int] = c.NUM_WORKERS,
    pin_memory: Optional[int] = c.PIN_MEMORY
) -> DataLoader:
    """Return pytorch dataloader for a given dataset.

    Args:
        dataset (Dataset): Pytorch dataset.
        batch_size (Optional[int], optional): Batch size.
        shuffle (Optional[bool], optional): Shuffle flag.
        num_workers (Optional[int], optional): Number of workers.
        pin_memory (Optional[int], optional): Pin memory flag.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory)
