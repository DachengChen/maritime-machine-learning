"""
Maritime Machine Learning – source package.

Files are stored with numeric prefixes (``01_``, ``02_``, …) for ordering.
This module re-registers each numbered file under its clean alias so that
existing imports such as ``from src.data_preprocessing import ...`` continue
to work without modification.
"""

import importlib.util as _ilu
import sys as _sys
import pathlib as _pl

_HERE = _pl.Path(__file__).parent

# Maps clean alias → numbered filename stem
_MODULE_MAP: dict[str, str] = {
    "data_preprocessing":   "01_data_preprocessing",
    "feature_engineering":  "02_feature_engineering",
    "eta_prediction":       "03_eta_prediction",
    "destination_prediction": "04_destination_prediction",
    "anomaly_detection":    "05_anomaly_detection",
    "model_evaluation":     "06_model_evaluation",
    "supervised_learning":  "07_supervised_learning",
    "unsupervised_learning": "08_unsupervised_learning",
    "meta_learning":        "09_meta_learning",
}

for _alias, _stem in _MODULE_MAP.items():
    _full_name = f"src.{_alias}"
    if _full_name not in _sys.modules:
        _spec = _ilu.spec_from_file_location(_full_name, _HERE / f"{_stem}.py")
        _mod = _ilu.module_from_spec(_spec)          # type: ignore[arg-type]
        _sys.modules[_full_name] = _mod
        _spec.loader.exec_module(_mod)               # type: ignore[union-attr]
