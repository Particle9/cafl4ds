# Experiment notebooks

Install the analysis environment from the repository root:

```powershell
..\.tools\Scripts\uv.exe sync --extra cpu --group analysis
..\.tools\Scripts\uv.exe run --extra cpu --group analysis jupyter lab
```

Start with [`01_visualize_existing_runs.ipynb`](01_visualize_existing_runs.ipynb). It discovers the newest canonical
health corpus under `outputs/getting-started/` or `outputs/adaptation/`, and plots accuracy, RankMe, and drift over
stream steps. P1.2 runs also receive matched current-versus-past condition plots. Edit `RUN_ROOT` in the configuration
cell to compare another run.

For a compact overview of the earlier P1.2 experiments, open
[`02_p12_experiment_history.ipynb`](02_p12_experiment_history.ipynb). It reads representative `comparison.json` files
from the saved STL-10 and BDD100K runs, then plots accuracy gains, backward transfer, forgetting, and the final
three-seed confirmation results. Missing runs are skipped, so the notebook still works with a partial output archive.
