"""Run M6.1.0, the paired learned-versus-fixed image-level top-4 MAE experiment."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from cafl4ds.cluster_routing_experiment import run_experiment


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="cluster_routing_experiment")  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Run or resume the two arms and save the complete scientific comparison."""
    run_experiment(cfg, Path(cfg.output))


if __name__ == "__main__":
    main()
