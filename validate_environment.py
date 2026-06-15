
"""Validate Python package dependencies for the XGBoost forecasting pipeline."""

from __future__ import annotations

import importlib
from importlib import metadata

REQUIRED_PACKAGES = {
    "joblib": ("joblib", "joblib>=1.3.2"),
    "matplotlib": ("matplotlib", "matplotlib>=3.8.0,<4.0"),
    "mlflow": ("mlflow", "mlflow>=3.0,<4.0"),
    "numpy": ("numpy", "numpy>=1.26.4,<3.0"),
    "pandas": ("pandas", "pandas>=2.1.0,<3.0"),
    "scikit-learn": ("sklearn", "scikit-learn>=1.4.0,<2.0"),
    "seaborn": ("seaborn", "seaborn>=0.13.0,<1.0"),
    "xgboost": ("xgboost", "xgboost>=2.0.0,<4.0"),
}


def main() -> int:
    """Import required packages, print versions, and report missing dependencies."""
    missing = []
    print("Validating XGBoost pipeline dependencies...")

    for distribution_name, (import_name, requirement_spec) in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
            version = metadata.version(distribution_name)
            print(f"OK      {distribution_name}=={version}")
        except (ImportError, metadata.PackageNotFoundError):
            missing.append(requirement_spec)
            print(f"MISSING {distribution_name} ({import_name})")

    if missing:
        install_command = "python -m pip install " + " ".join(
            f'"{requirement}"' for requirement in missing
        )
        print("\nMissing packages detected.")
        print(f"Install command:\n{install_command}")
        return 1

    print("\nAll required packages are installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
