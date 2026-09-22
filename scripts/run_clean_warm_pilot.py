"""Run the two predeclared clean-warm BDD100K pilot arms with one shared decode."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from scripts.run_adaptation_experiment import run_experiment

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _ROOT / "cafl4ds" / "configs"
_OUTPUT_ROOT = _ROOT / "outputs" / "adaptation-bdd" / "clean-warm-mae" / "20260921-pilot-s101"


def main() -> None:
    """Execute full-backbone and last-block arms from the same clean warm checkpoint."""
    arms = (("full", 1e-4), ("last_block", 3e-4))
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        for scope, learning_rate in arms:
            config = compose(
                config_name="adaptation_bdd_clean_warm_mae",
                overrides=[f"train_scope={scope}", f"optim.lr={learning_rate}"],
            )
            config.run_name = f"bdd_clean_warm_mae_{scope}"
            run_experiment(config, _OUTPUT_ROOT / scope)


if __name__ == "__main__":
    main()
