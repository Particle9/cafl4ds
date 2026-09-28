"""Hydra CLI for P1.4.0 CPU experiment stages."""

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from cafl4ds.collapse_diet import run_stage
from cafl4ds.jsonio import dumps_valid


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="collapse_diet")
def main(config: DictConfig) -> None:
    """Run a stage and write the result into Hydra's output directory too."""
    summary = run_stage(config)
    Path(HydraConfig.get().runtime.output_dir, "stage_summary.json").write_text(dumps_valid(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
