import ast
from collections.abc import Callable
import os
from typing import Optional, Tuple


import pandas as pd
from pandas.api.types import is_string_dtype
from PIL import Image
import torch
from torch.utils.data import Dataset

from utils import pil_loader
import constants as c


class CSVDataset(Dataset):
    """Custom dataset that loads images and targets from paths in csv.
    """

    def __init__(
            self,
            csv_file: pd.DataFrame,
            path_col: str,
            target_col: str,
            root_dir: str,
            transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
            explode: bool = False):
        """
        Args:
            csv_file (string): Path to the csv file.
            path_col (string): Column name for relative paths.
            target_col (string): Column name with targets.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
              on a sample.
            explode (bool, optional): Optional option to explode target_col
              into separate rows.
        """
        self.csv_df = pd.read_csv(csv_file)
        self.path_col = path_col
        self.target_col = target_col
        self.root_dir = root_dir
        self.transform = transform

        if is_string_dtype(self.csv_df[target_col]):
            self.csv_df[target_col] = self.csv_df[target_col].apply(
                lambda x: ast.literal_eval(x))

        if explode:
            self.csv_df = self.csv_df.explode(
                self.target_col).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.csv_df)

    def __getitem__(self, idx) -> Tuple[Image.Image, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(
            self.root_dir, self.csv_df.iloc[idx][self.path_col])
        image = pil_loader(img_path)

        if self.transform:
            image = self.transform(image)

        target = self.csv_df.iloc[idx][self.target_col]
        return image, torch.tensor(target)


def get_csv_dataset(
    csv_file: str,
    root_dir: str,
    train: Optional[bool] = False,
    path_col: Optional[str] = c.PATH_COL,
    target_col: Optional[str] = c.TARGET_COL,
) -> Dataset:
    """Return pytorch dataset for Dollar Street csv.

    Args:
        csv_file (str): Path to the csv file.
        root_dir (str): Directory with all the images.
        train (Optional[bool], optional): Train flag (i.e. train or val).
        path_col (Optional[str], optional): Column name for relative paths.
        target_col (Optional[str, optional): Column name with targets.

    Returns:
        Dataset: CSVDataset for given file.
    """
    pass
