"""Explore M6.1.0. Run: marimo edit notebooks/m6_1_0.py.

Set M6_1_ARTIFACT_DIR to inspect another completed run, such as the short pilot.
"""

# mypy: ignore-errors
# marimo generates cell signatures and module-valued dependency arguments.

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from cafl4ds.cluster_routing_analysis import (
        COLORS,
        plot_learning,
        plot_paired,
        plot_regimes,
        plot_routing,
        read_rows,
    )

    return COLORS, Path, json, mo, np, os, plot_learning, plot_paired, plot_regimes, plot_routing, plt, read_rows


@app.cell
def _(Path, json, os):
    output = Path(os.environ.get("M6_1_ARTIFACT_DIR", str(Path(__file__).resolve().parents[1] / "outputs/m6_1_0")))
    comparison = json.loads((output / "comparison.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    membership = json.loads((output / "membership.json").read_text())
    seeds = protocol["config"]["seeds"]
    layer_ids = protocol["config"]["encoder"].get("moe_blocks") or list(
        range(max(0, protocol["config"]["encoder"]["depth"] - 2), protocol["config"]["encoder"]["depth"])
    )
    return comparison, layer_ids, membership, output, protocol, seeds


@app.cell
def _(comparison, mo, protocol, seeds):
    _run = comparison["runs"][0]
    mo.md(
        f"""
        # M6.1.0 — learned versus fixed GMM image routing

        **{len(seeds)} paired seeds · {_run["steps"]:,} updates per arm · batch {protocol["config"]["batch_size"]}**

        Both models use a small, randomly initialized MAE: 64-pixel images, four encoder blocks,
        four experts in blocks 2/3, 75% masking, and a dense decoder. **All four experts execute**,
        weighted by one probability vector per image. Shared parameters, image order, augmentations
        and masks are matched within each seed. All training images are accepted.

        **Learned:** mean visible-token representation (including CLS), an independent gate at each
        MoE layer, training logit noise and importance balancing. **GMM:** the original image's fixed
        M6.0.0 probabilities, reused across every layer, crop and mask; no trainable router.
        The GMM deliberately saw all BDD images when fitted.

        Each pass follows the same 18 time-of-day × weather blocks. Standard health measurements
        run every 50 updates, with extra reads at block exits. Per-regime reconstruction uses fixed
        official-validation images and four fixed view/mask draws. Some rare regimes have very
        small panels; sample counts appear below.
        """
    )
    return


@app.cell
def _(comparison, mo):
    _names = {
        "heldout_mse": "Final natural held-out MSE",
        "pass_1_end_minus_exit": "Pass 1: mean MSE deterioration after block exit",
        "pass_2_end_minus_exit": "Pass 2: mean MSE deterioration after block exit",
        "balanced_linear": "Final balanced scene linear probe",
        "balanced_knn": "Final balanced scene kNN",
        "rankme": "Final RankMe",
    }
    _rows = []
    for _key, _name in _names.items():
        if _key in comparison["gmm_minus_learned"]:
            _left, _right = (comparison["arms"][_arm][_key] for _arm in ("learned", "gmm"))
            _effect = comparison["gmm_minus_learned"][_key]
            _rows.append(
                {
                    "Measure": _name,
                    "Learned (seed mean)": round(_left["mean"], 5),
                    "GMM (seed mean)": round(_right["mean"], 5),
                    "GMM − learned": round(_effect["mean"], 5),
                    "Paired 95% interval": str([round(_v, 5) for _v in _effect["ci95"]])
                    if _effect["ci95"]
                    else "One seed",
                }
            )
    mo.vstack(
        [
            mo.md(
                "## Endpoints and paired differences\n"
                "Lower MSE and deterioration are better; higher probe accuracy is better. "
                "Deterioration is each regime's MSE at pass end minus MSE just after its own block, "
                "averaged across regimes with subsequent updates. Negative values mean improvement. "
                "Intervals use training seeds as replicates; these are descriptive comparisons."
            ),
            mo.ui.table(_rows, selection=None),
        ]
    )
    return


@app.cell
def _(comparison, mo):
    _effect = comparison["gmm_minus_learned"]["heldout_mse"]
    _relative = 100 * (
        comparison["arms"]["gmm"]["heldout_mse"]["mean"] / comparison["arms"]["learned"]["heldout_mse"]["mean"] - 1
    )
    _interval = _effect["ci95"]
    _uncertainty = (
        "The descriptive paired interval includes zero, so endpoint reconstruction does not establish an advantage."
        if _interval and _interval[0] <= 0 <= _interval[1]
        else "Inspect the individual seed pairs and the other outcomes before drawing an overall conclusion."
    )
    mo.md(
        f"**Endpoint reconstruction:** GMM differs from learned routing by **{_relative:+.2f}%** in mean MSE. "
        f"{_uncertainty} The routing/gradient plots below show whether the strategies distribute learning differently; "
        "a structured routing pattern by itself does not establish better retention or representations."
    )
    return


@app.cell
def _(comparison, mo, plot_paired):
    mo.vstack([mo.md("### Individual seed pairs"), plot_paired(comparison)])
    return


@app.cell
def _(mo, output, plot_learning):
    mo.vstack([mo.md("## Learning and standard health signals"), plot_learning(output)])
    return


@app.cell
def _(mo):
    health_metric = mo.ui.dropdown(
        options={
            "Uniformity": "uniformity",
            "Directional feature variance": "mean_feature_var",
            "Off-diagonal covariance": "offdiag_cov",
            "Cosine drift": "cosine_drift",
            "Balanced reconstruction MSE": "balanced_mse",
            "Balanced scene kNN": "balanced_knn",
            "Natural scene linear probe": "natural_linear",
            "Natural scene kNN": "natural_knn",
        },
        value="Uniformity",
        label="Additional health signal",
    )
    health_metric
    return (health_metric,)


@app.cell
def _(COLORS, health_metric, np, output, plt, read_rows, seeds):
    _fig, _ax = plt.subplots(figsize=(13, 4), constrained_layout=True)
    for _arm in ("learned", "gmm"):
        _maps = [
            {
                _row["step"]: _row[health_metric.value]
                for _row in read_rows(output / f"seed_{_seed}" / _arm / "evaluation.jsonl")
                if health_metric.value in _row
            }
            for _seed in seeds
        ]
        _steps = sorted(set.intersection(*(set(_values) for _values in _maps)))
        _matrix = np.array([[_values[_step] for _step in _steps] for _values in _maps])
        _mean = _matrix.mean(0)
        _sem = _matrix.std(0, ddof=1) / np.sqrt(len(seeds)) if len(seeds) > 1 else np.zeros_like(_mean)
        _ax.plot(_steps, _mean, color=COLORS[_arm], label=_arm)
        _ax.fill_between(_steps, _mean - _sem, _mean + _sem, color=COLORS[_arm], alpha=0.15)
    _ax.set(title=f"{health_metric.value} · seed mean ±1 SEM", xlabel="Gradient update")
    _ax.legend()
    _ax.grid(alpha=0.2)
    _fig
    return


@app.cell
def _(layer_ids, membership, mo, seeds):
    seed_control = mo.ui.dropdown(options={f"Seed {s}": s for s in seeds}, value=f"Seed {seeds[0]}", label="Seed")
    layer_control = mo.ui.dropdown(
        options={f"Layer {layer}": layer for layer in layer_ids}, value=f"Layer {layer_ids[0]}", label="MoE layer"
    )
    regime_mode = mo.ui.radio(
        options={"Raw normalized MSE": "raw", "Change since block exit": "change"},
        value="Raw normalized MSE",
        label="Reconstruction heatmap",
    )
    regime_control = mo.ui.dropdown(
        options={f"{r}: {name}": r for r, name in membership["regime_names"].items()},
        value=next(iter(f"{r}: {name}" for r, name in membership["regime_names"].items())),
        label="Regime for detailed trajectory",
    )
    mo.hstack([seed_control, layer_control, regime_mode, regime_control], wrap=True)
    return layer_control, regime_control, regime_mode, seed_control


@app.cell
def _(mo, output, plot_regimes, regime_mode, seed_control):
    mo.vstack(
        [
            mo.md(
                "## Per-regime reconstruction and retention\n"
                "The two arms share a color scale. Change-since-exit values restart each pass and are blank "
                "before that regime's block exits; positive values indicate worse reconstruction."
            ),
            plot_regimes(output, seed_control.value, change_since_exit=regime_mode.value == "change"),
        ]
    )
    return


@app.cell
def _(COLORS, json, membership, mo, output, plt, read_rows, regime_control, seed_control):
    _fig, _ax = plt.subplots(figsize=(13, 4), constrained_layout=True)
    for _arm in ("learned", "gmm"):
        _rows = read_rows(output / f"seed_{seed_control.value}" / _arm / "evaluation.jsonl")
        _ax.plot(
            [r["step"] for r in _rows],
            [r["regime_mse"][regime_control.value] for r in _rows],
            color=COLORS[_arm],
            label=_arm,
        )
    _result = json.loads((output / f"seed_{seed_control.value}/learned/complete.json").read_text())
    for _entry in _result["forgetting"]:
        if str(_entry["regime"]) == regime_control.value:
            _ax.axvspan(_entry["start_step"], _entry["exit_step"], color="gray", alpha=0.16)
    _ax.set(
        title=f"{membership['regime_names'][regime_control.value]} · seed {seed_control.value}",
        xlabel="Gradient update",
        ylabel="Fixed-panel normalized masked MSE",
    )
    _ax.legend()
    _ax.grid(alpha=0.2)
    mo.vstack([mo.md("Gray spans mark when this regime is being trained."), _fig])
    return


@app.cell
def _(layer_control, mo, output, plot_routing, seed_control):
    mo.vstack(
        [
            mo.md(
                "## Which experts receive weight and gradients?\n"
                "Top-1 is **argmax of the minibatch-mean probability vector**, recorded from the actual "
                "training forward, including learned-gate noise. It is a summary of weighting: all four "
                "experts execute. Unclipped gradient norms show where task updates concentrate; AdamW "
                "momentum and weight decay can also move parameters. Expert IDs are shown within the "
                "selected seed; learned expert identities are not averaged across seeds."
            ),
            plot_routing(output, seed_control.value, layer_control.value),
        ]
    )
    return


@app.cell
def _(layer_control, membership, mo, np, output, plt, read_rows, seed_control):
    _fig, _axes = plt.subplots(2, 1, figsize=(15, 6), constrained_layout=True, sharex=True)
    _regimes = sorted(map(int, membership["regime_names"]))
    for _ax, _arm in zip(_axes, ("learned", "gmm"), strict=True):
        _mass, _counts = np.zeros((4, len(_regimes))), np.zeros(len(_regimes))
        for _row in read_rows(output / f"seed_{seed_control.value}" / _arm / "training.jsonl"):
            for _regime, _values in _row[f"l{layer_control.value}_mass_by_regime"].items():
                _column = _regimes.index(int(_regime))
                _mass[:, _column] += np.array(_values)
                _counts[_column] += _row["regime_counts"][int(_regime)]
        _mesh = _ax.imshow(_mass / np.maximum(_counts, 1)[None, :], aspect="auto", vmin=0, vmax=1, cmap="magma")
        _ax.set(
            title=f"{_arm} · layer {layer_control.value} · seed {seed_control.value}",
            ylabel="Expert",
            yticks=range(4),
            xticks=range(len(_regimes)),
        )
    _axes[-1].set_xticklabels(
        [membership["regime_names"][str(r)] for r in _regimes], rotation=50, ha="right", fontsize=8
    )
    _fig.colorbar(_mesh, ax=list(_axes), label="Mean training probability")
    mo.vstack(
        [
            mo.md(
                "### Expert probability mass by actual image regime\n"
                "These values use per-image regime sums, including exact attribution inside mixed boundary batches."
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(membership, mo):
    _counts = {}
    for _regime in membership["regime_eval_regimes"]:
        _counts[_regime] = _counts.get(_regime, 0) + 1
    mo.ui.table(
        [
            {"Regime ID": int(_r), "Regime": _name, "Validation images": _counts.get(int(_r), 0)}
            for _r, _name in membership["regime_names"].items()
        ],
        selection=None,
    )
    return


if __name__ == "__main__":
    app.run()
