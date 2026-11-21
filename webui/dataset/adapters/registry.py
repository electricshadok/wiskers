"""
Central adapter registry and display-name formatting for WebUI dataset configs.
"""

from pathlib import Path

from webui.dataset.adapters.adroit_hand import AdroitHandUIAdapter
from webui.dataset.adapters.base import DefaultUIAdapter, EnvUIAdapter
from webui.dataset.adapters.breakout import BreakoutUIAdapter
from webui.dataset.adapters.car_racing import CarRacingUIAdapter
from webui.dataset.adapters.franka_kitchen import FrankaKitchenUIAdapter
from webui.dataset.adapters.inverted_pendulum import InvertedPendulumUIAdapter
from webui.dataset.adapters.shadow_hand import ShadowHandUIAdapter
from wiskers.data.datamodule import TrajectoryDataModule


_ADAPTERS: tuple[type[EnvUIAdapter], ...] = (
    AdroitHandUIAdapter,
    BreakoutUIAdapter,
    CarRacingUIAdapter,
    FrankaKitchenUIAdapter,
    InvertedPendulumUIAdapter,
    ShadowHandUIAdapter,
)
_BY_ENV_NAME: dict[str, type[EnvUIAdapter]] = {cls.env_name: cls for cls in _ADAPTERS}
_BY_CONFIG_STEM: dict[str, type[EnvUIAdapter]] = {cls.config_stem: cls for cls in _ADAPTERS}


def get_adapter(data_module: TrajectoryDataModule) -> EnvUIAdapter:
    """Return the UI adapter matching ``data_module.collector.env``."""
    adapter_cls = _BY_ENV_NAME.get(type(data_module.collector.env).__name__, DefaultUIAdapter)
    return adapter_cls()


def format_config_name(path: str) -> str:
    """Format a dataset config path into a descriptive UI dropdown label."""
    stem = Path(path).stem.lower()
    adapter_cls = _BY_CONFIG_STEM.get(stem)
    label = adapter_cls.display_name if adapter_cls is not None else stem.replace("_", " ").title()
    return f"{label} ({Path(path).name})"
