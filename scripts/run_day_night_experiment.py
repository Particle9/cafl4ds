"""Run the paired M6.3.0 day+dusk to night MAE experiment."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from cafl4ds.day_night_experiment import run_experiment


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="day_night_experiment")  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Run or resume the experiment and verify the paired traces."""
    run_experiment(cfg, Path(cfg.output))


if __name__ == "__main__":
    main()
