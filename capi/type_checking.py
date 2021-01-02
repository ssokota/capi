"""Functions for runtime type checking. More strict but slower than availabe
static type checking. Off by default.
"""
import os
from typing import Any, Optional, Tuple

import torch


def assert_joint_probability(
    x: torch.Tensor, shape: Tuple[int, ...], allow_improper: bool = False
) -> None:
    """Assert `x` is joint probability distribution over two variables

    Args:
        x: Possible joint probability distribution
        shape: Required shape
        allow_improper: Whether improper distribution (all zeros) is permitted
    """
    if os.getenv("strict_type_check") == "1":
        norm = x.sum(dim=(-1, -2))
        if allow_improper:
            norm[norm == 0] = 1
        assert torch.isclose(norm, torch.Tensor([1.0]).to(norm.device)).all()
        assert x.shape[-1] == x.shape[-2]
        assert x.shape == shape


def assert_prescription(
    x: torch.Tensor,
    shape: Tuple[int, ...],
    pure: bool = True,
    allow_improper: bool = False,
) -> None:
    """Assert `x` is valid prescription

    Args:
        x: Possible prescription
        shape: Required shape
        pure: Whether prescription is required to be deterministic
        allow_improper: Whether improper distribution (all zeros) is permitted
    """
    if os.getenv("strict_type_check") == "1":
        norm = x.sum(dim=-1)
        if allow_improper:
            norm[norm == 0] = 1
        assert torch.isclose(norm, torch.Tensor([1.0]).to(x.device)).all()
        assert (x >= 0).all()
        assert x.shape == shape
        if pure:
            max_vals = x.max(dim=-1).values
            if allow_improper:
                max_vals[max_vals == 0] = 1
            assert (max_vals == 1).all()


def assert_label_prescription(
    x: torch.Tensor, num_actions: int, shape: Tuple[int, ...]
) -> None:
    """Assert `x` is valid label prescription

    Args:
        x: Possible prescription
        num_actions: Number of action labels
        shape: Required shape
    """
    if os.getenv("strict_type_check") == "1":
        assert x.dtype == torch.int64
        assert (x >= 0).all()
        assert (x < num_actions).all()
        assert x.shape == shape


def assert_shape(
    x: torch.Tensor, shape: Tuple[int, ...], dim: Optional[int] = None
) -> None:
    """Assert `x` has shape `shape`

    Args:
        x: Tensor
        shape: Required shape
        dim: If specified, enforce shape requirement only for axis `dim`
    """
    if os.getenv("strict_type_check") == "1":
        if dim:
            assert (x.shape[dim],) == shape
        else:
            assert x.shape == shape


def assert_num_dims(x: torch.Tensor, num_dims: int) -> None:
    """Assert `x` has `num_dims` dimensions

    Args:
        x: Tensor
        num_dims: Required number of dimensions
    """
    if os.getenv("strict_type_check") == "1":
        assert len(x.shape) == num_dims


def assert_element(x: Any, collection: Tuple[Any, ...]) -> None:
    """Assert `x` in `collection`

    Args:
        x: Anything
        collection: Tuple
    """
    if os.getenv("strict_type_check") == "1":
        assert x in collection
