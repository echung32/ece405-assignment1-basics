from typing import Tuple
import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Int


def get_batch(
        dataset: npt.NDArray,
        batch_size: int,
        context_length: int,
        device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """

    n = len(dataset) - context_length

    indices = np.random.randint(
        low=0,
        high=n,
        size=batch_size,
    )

    # the sampled input sequences
    inputs: Int[np.ndarray, "batch_size context_length"] = np.stack([
        dataset[i:i + context_length]
        for i in indices
    ])

    # the corresponding next-token targets. (input shifted by 1)
    targets: Int[np.ndarray, "batch_size context_length"] = np.stack([
        dataset[i + 1:i + 1 + context_length]
        for i in indices
    ])

    # move inputs to the device and convert to long
    # cause it says, "Tuple of torch.LongTensors of shape" in the adapter
    inputs_tensor: Int[torch.Tensor, "batch_size context_length"] = torch.from_numpy(inputs).long().to(device)
    targets_tensor: Int[torch.Tensor, "batch_size context_length"] = torch.from_numpy(targets).long().to(device)

    return inputs_tensor, targets_tensor
