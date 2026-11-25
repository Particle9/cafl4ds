# Research notebooks

| Topic | Notebooks | Coverage |
| -- | -: | -- |
| [Foundations](selected/foundations/) | 3 | Early SSL filtering, representation-health measurements, and forecast baselines. |
| [Single-image admission](selected/single-image-admission/) | 16 | Image-level filtering, online objective prediction, and learning-rate choices. |
| [Transition models and MPC](selected/transition-models-and-mpc/) | 38 | Action-response forecasts, rate rollouts, paired plans, and predictor benchmarks. |
| [Capability and action response](selected/capability-and-response/) | 20 | Representation retention, delayed benefit, and budget-matched adaptation. |
| [Health-guided replay](selected/health-guided-replay/) | 14 | Replay composition, memory banks, and health-constrained update allocation. |
| [O3 control](selected/o3-control/) | 37 | O3 action selection, replay controls, distribution shifts, and model-component allocation. |
| [DINO](selected/dino/) | 23 | Teacher and student timescales, adaptation depth, and semantic acquisition. |
| [Masked autoencoders](selected/mae/) | 37 | MAE reconstruction, geometry, counterfactual forecasts, and multi-step control. |
| [SimSiam and temporal learning](selected/simsiam-and-temporal/) | 16 | SimSiam objectives, within-clip forecasts, and object correspondence. |
| [Semantic adaptation](selected/semantic-adaptation/) | 23 | Learnability, retention, replay attribution, and transfer across regimes. |

Each topic index lists the notebooks by experiment ID or filename. Keep the experiment ID in the filename when adding a
notebook.

## Visualizing saved runs

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
