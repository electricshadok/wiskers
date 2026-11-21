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
        BoundedActionSpec(low=torch.zeros(2, 3), high=torch.ones(2, 3))
    with pytest.raises(ValueError):
        BoundedActionSpec(low=torch.zeros(0), high=torch.ones(0))
    with pytest.raises(ValueError):
        BoundedActionSpec(low=torch.zeros(2), high=torch.ones(3))
    with pytest.raises(ValueError):
        BoundedActionSpec(low=torch.zeros(3, dtype=torch.float32), high=torch.ones(3, dtype=torch.float64))
    with pytest.raises(ValueError, match="floating-point"):
        BoundedActionSpec(low=torch.zeros(3, dtype=torch.int64), high=torch.ones(3, dtype=torch.int64))
    with pytest.raises(ValueError, match="floating-point"):
        BoundedActionSpec(low=torch.zeros(3, dtype=torch.bool), high=torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError, match="devices"):
        BoundedActionSpec(low=torch.zeros(2), high=torch.ones(2, device="meta"))
    with pytest.raises(ValueError, match="finite"):
        BoundedActionSpec(low=torch.tensor([float("nan")]), high=torch.tensor([1.0]))
    with pytest.raises(ValueError, match="finite"):
        BoundedActionSpec(low=torch.tensor([float("-inf")]), high=torch.tensor([1.0]))
    with pytest.raises(ValueError, match="finite"):
        BoundedActionSpec(low=torch.tensor([0.0]), high=torch.tensor([float("inf")]))
    with pytest.raises(ValueError):
        BoundedActionSpec(low=torch.full((3,), 2.0), high=torch.full((3,), 1.0))


def test_bounded_action_spec_device():
    cpu_spec = BoundedActionSpec(
        low=torch.tensor([-1.0, 0.0, -0.5]),
        high=torch.tensor([1.0, 2.0, 0.5]),
    )
    assert cpu_spec.sample().device == cpu_spec.low.device

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else None)
    if device is not None:
        dev_spec = BoundedActionSpec(
            low=cpu_spec.low.to(device),
            high=cpu_spec.high.to(device),
        )
        assert cpu_spec != dev_spec
        sample = dev_spec.sample()
        assert sample.device == dev_spec.low.device
        assert dev_spec.contains(sample)
