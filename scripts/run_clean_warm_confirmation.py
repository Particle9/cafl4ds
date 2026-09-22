"""Run the fresh-seed P1.2 confirmation for full adaptation and compute-matched replay."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from scripts.run_adaptation_experiment import run_experiment

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _ROOT / "cafl4ds" / "configs"
_OUTPUT_ROOT = _ROOT / "outputs" / "adaptation-bdd" / "clean-warm-confirmation" / "20260922"
_SEEDS = [103, 107, 109]


def _run(config_name: str, output_name: str) -> None:
    """Compose one arm at the confirmation seeds and write its ensemble artifacts."""
    config = compose(config_name=config_name, overrides=[f"seeds={_SEEDS}"])
    run_experiment(config, _OUTPUT_ROOT / output_name)


def main() -> None:
    """Run both confirmation arms in one process so their decoded sources are shared."""
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        _run("adaptation_bdd_clean_warm_mae", "full")
        _run("adaptation_bdd_clean_warm_mae_replay", "replay")


if __name__ == "__main__":
    main()
