import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PEPTIDE_LENGTH = 14
AA_TO_INDEX = {
    aa: i for i, aa in enumerate("LAGVESIKRDTPNFQYHMCW")
}  # dictionary of amino acids to index
INDEX_TO_AA = dict(
    enumerate("LAGVESIKRDTPNFQYHMCW")
)  # dictionary of index to amino acids


def fold_dataloaders(
    folds: list[pd.DataFrame], batch_size: int = 32, num_workers: int = 7
) -> list[tuple[DataLoader, DataLoader]]:
    """Create train and validation dataloaders for each fold.

    Args:
        folds (list[pd.DataFrame]): List of folds.
        batch_size (int, optional): Batch size. Defaults to 32.
        num_workers (int, optional): Number of workers. Defaults to 7.

    Returns:
        list[tuple[DataLoader, DataLoader]]: List of train and validation dataloaders for each fold.
    """

    dataloaders = []

    for fold in range(len(folds)):
        train_df = pd.concat(folds[:fold] + folds[fold + 1 :])
        val_df = folds[fold]

        train_dataset = PeptideDataset(train_df)
        val_dataset = PeptideDataset(val_df)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=num_workers,
        )

        dataloaders.append((train_dataloader, val_dataloader))

    return dataloaders


def get_test_dataloader(
    test_df: pd.DataFrame, batch_size: int = 32, num_workers: int = 7
) -> DataLoader:
    """Create test dataloader.

    Args:
        test_df (pd.DataFrame): Test dataframe.
        batch_size (int, optional): Batch size. Defaults to 32.
        num_workers (int, optional): Number of workers. Defaults to 7.

    Returns:
        DataLoader: Test dataloader.
    """

    test_dataset = PeptideDataset(test_df)

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )


class PeptideDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        """Initialize the dataset.

        Args:
            dataframe (pd.DataFrame): Dataframe containing the peptide and target.
        """
        peptide = dataframe["peptide"]
        self.peptide = peptide.apply(lambda x: np.array([AA_TO_INDEX[aa] for aa in x]))  # type: ignore
        self.target = dataframe["hit"]

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        pep = torch.tensor(self.peptide.iloc[idx], dtype=torch.long)
        pep = torch.nn.functional.pad(
            pep, (0, PEPTIDE_LENGTH - len(pep)), value=len(AA_TO_INDEX)
        )
        return pep, torch.tensor(self.target.iloc[idx], dtype=torch.float)
