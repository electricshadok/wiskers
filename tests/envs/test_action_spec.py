from dataclasses import FrozenInstanceError

import pytest
import torch

from wiskers.envs.action_spec import ActionSpec, BoundedActionSpec, DiscreteActionSpec


def test_discrete_action_spec():
    spec = DiscreteActionSpec(n=4)
    assert isinstance(spec, ActionSpec)
    assert spec.action_dim == 1
    assert spec.shape == (1,)
    assert spec.dtype == torch.int64

    sample = spec.sample()
    assert spec.contains(sample)
    assert not spec.contains(torch.tensor([4], dtype=torch.int64))
    assert not spec.contains(torch.tensor([-1], dtype=torch.int64))
    assert not spec.contains(torch.tensor([1.0], dtype=torch.float32))
    assert not spec.contains(torch.tensor(1, dtype=torch.int64))

    with pytest.raises(FrozenInstanceError):
        setattr(spec, "n", 2)
    with pytest.raises(ValueError):
        DiscreteActionSpec(n=0)
    with pytest.raises(ValueError):
        DiscreteActionSpec(n=4, dtype=torch.float32)


def test_bounded_action_spec_float():
    spec = BoundedActionSpec(
        low=torch.full((3,), -2.0),
        high=torch.full((3,), 1.0),
    )
    assert isinstance(spec, ActionSpec)
    assert spec.action_dim == 3
    assert spec.shape == (3,)
    assert spec.dtype == torch.float32
    assert spec == BoundedActionSpec(low=torch.full((3,), -2.0), high=torch.full((3,), 1.0))
    assert spec != BoundedActionSpec(low=torch.full((3,), -1.0), high=torch.full((3,), 1.0))

    sample = spec.sample()
    assert spec.contains(sample)
    assert not spec.contains(torch.tensor([0.0, 0.0, 2.0], dtype=torch.float32))
    assert not spec.contains(torch.tensor([-3.0, 0.0, 0.0], dtype=torch.float32))
    assert not spec.contains(torch.tensor([0, 0, 0], dtype=torch.int64))
    assert not spec.contains(torch.tensor([0.0, 0.0], dtype=torch.float32))

    with pytest.raises(FrozenInstanceError):
        setattr(spec, "low", torch.zeros(3))


def test_bounded_action_spec_validation():
    with pytest.raises(ValueError):
        BoundedActionSpec(low=torch.tensor(0.0), high=torch.tensor(1.0))
    with pytest.raises(ValueError):
        BoundedActionSpec(low=torch.zeros(2), high=torch.ones(3))
    with pytest.raises(ValueError):
        BoundedActionSpec(low=torch.zeros(3, dtype=torch.float32), high=torch.ones(3, dtype=torch.int64))
    with pytest.raises(ValueError):
        BoundedActionSpec(low=torch.zeros(3, dtype=torch.bool), high=torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError):
        BoundedActionSpec(low=torch.full((3,), 2.0), high=torch.full((3,), 1.0))


def test_bounded_action_spec_int():
    spec = BoundedActionSpec(
        low=torch.tensor([0, -1, 2], dtype=torch.int64),
        high=torch.tensor([2, 1, 4], dtype=torch.int64),
    )
    assert isinstance(spec, ActionSpec)
    assert spec.action_dim == 3
    assert spec.shape == (3,)
    assert spec.dtype == torch.int64

    for _ in range(20):
        sample = spec.sample()
        assert spec.contains(sample)
    assert not spec.contains(torch.tensor([3, 0, 2], dtype=torch.int64))


def test_bounded_action_spec_unbounded():
    spec = BoundedActionSpec(
        low=torch.full((2,), float("-inf")),
        high=torch.full((2,), float("inf")),
    )
    sample = spec.sample()
    assert spec.contains(sample)
    assert torch.isfinite(sample).all()
