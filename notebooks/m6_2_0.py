"""Explore M6.2.0. Run: marimo edit notebooks/m6_2_0.py.

Set M6_2_ARTIFACT_DIR to inspect another completed run, such as the short pilot.
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
        plot_routing,
        read_rows,
    )

    return COLORS, Path, json, mo, np, os, plot_learning, plot_paired, plot_routing, plt, read_rows


@app.cell
def _(Path, json, os):
    output = Path(os.environ.get("M6_2_ARTIFACT_DIR", str(Path(__file__).resolve().parents[1] / "outputs/m6_2_0")))
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
        # M6.2.0 — expert-specific learning rates

        **{len(seeds)} paired seeds · {_run["steps"]:,} updates per arm · batch {protocol["config"]["batch_size"]}**

        The original capped arms use a small, randomly initialized MAE: 64-pixel images, four encoder blocks,
        four experts in blocks 2/3, 75% masking, and a dense decoder. **All four experts execute**,
        weighted by one probability vector per image. Shared parameters, image order, augmentations
        and masks are matched within each seed. All training images are accepted.

        **Learned:** mean visible-token representation (including CLS), an independent gate at each
        MoE layer, training logit noise and importance balancing. **GMM:** the original image's fixed
        M6.0.0 probabilities, reused across every layer, crop and mask; no trainable router.
        The GMM deliberately saw all BDD images when fitted.

        **Original intervention:** in both capped arms, each expert's AdamW learning rate is
        `0.001 × min(1, 4 × q_e)`, where `q_e` is that layer's actual minibatch-mean
        probability, including learned-gate noise. Uniform routing retains the base rate;
        an expert with 1% mass gets a rate of 0.00004. The cap prevents rates above 0.001.
        Shared parameters and learned routers retain 0.001. Moments, counters, clipping and
        losses follow the original experiment. AdamW weight decay uses the scaled learning
        rate too; its coefficient remains 0.05. No momentum resets or skipped expert steps.

        **Additional GMM arm:** `s_e = 4 × q_e` with no cap, so expert learning rates can reach
        0.004, and the four expert rates sum to approximately 0.004 per layer per step.
        Shared parameters retain 0.001. Its five seeds reuse the original GMM initial tensors,
        images, augmentations and masks. The original capped arms are retained as controls.
        The three-arm views below compare all conditions; the older two-arm sections are labeled capped.

        Each pass follows the same 18 time-of-day × weather blocks. Standard health measurements
        run every 50 updates, with extra reads at block exits. Per-regime reconstruction uses fixed
        official-validation images and four fixed view/mask draws. Some rare regimes have very
        small panels; sample counts appear below.
        """
    )
    return


@app.cell
def _(Path, json, mo, os, output):
    from cafl4ds.gmm_uncapped_analysis import (
        CONDITIONS,
        condition_folders,
        plot_added_regimes,
        plot_conditions,
        plot_three_rates,
    )

    uncapped_output = Path(os.environ.get("M6_2_UNCAPPED_ARTIFACT_DIR", str(output.parent / "m6_2_0_gmm_uncapped")))
    mo.stop(not (uncapped_output / "comparison.json").exists(), mo.md("The added GMM arm is still running."))
    uncapped_comparison = json.loads((uncapped_output / "comparison.json").read_text())
    return (
        CONDITIONS,
        condition_folders,
        plot_added_regimes,
        plot_conditions,
        plot_three_rates,
        uncapped_comparison,
        uncapped_output,
    )


@app.cell
def _(CONDITIONS, mo, output, plot_conditions, uncapped_comparison, uncapped_output):
    _metrics = ("heldout_mse", "time_mean_mse_pass_1", "pass_2_end_minus_exit", "rankme", "balanced_linear")
    _means = [
        {
            "Metric": _key,
            **{
                CONDITIONS[_condition][0]: round(_stats[_key]["mean"], 6)
                for _condition, _stats in uncapped_comparison["conditions"].items()
            },
        }
        for _key in _metrics
    ]
    _effects = []
    for _control, _values in uncapped_comparison["uncapped_minus"].items():
        for _key in _metrics:
            _effect = _values[_key]
            _effects.append(
                {
                    "Control": CONDITIONS[_control][0],
                    "Metric": _key,
                    "GMM 4q − control": round(_effect["mean"], 6),
                    "Paired 95% interval": str([round(_x, 6) for _x in _effect["ci95"]])
                    if _effect["ci95"]
                    else "One seed",
                }
            )
    mo.vstack(
        [
            mo.md(
                "## Additional GMM arm: uncapped 4q\n"
                "The capped learned and GMM arms are the completed M6.2.0 runs. Only the uncapped GMM arm "
                "is newly trained. Every new initial tensor, image order, augmentation, mask and fixed-routing "
                "readout was checked against the capped GMM arm. Intervals pair the same five seeds; "
                "lower MSE/deterioration and higher RankMe/probe accuracy are better."
            ),
            mo.ui.table(_means, selection=None),
            mo.ui.table(_effects, selection=None),
            plot_conditions(uncapped_output, output),
        ]
    )
    return


@app.cell
def _(mo, uncapped_comparison):
    _stats = uncapped_comparison["conditions"]
    _capped = _stats["gmm_capped"]
    _uncapped = _stats["gmm_uncapped"]
    _mse_percent = 100 * (_uncapped["heldout_mse"]["mean"] / _capped["heldout_mse"]["mean"] - 1)
    _probe_pp = 100 * (_uncapped["balanced_linear"]["mean"] - _capped["balanced_linear"]["mean"])
    mo.md(
        f"**Uncapped versus capped GMM:** final MSE changes by **{_mse_percent:+.2f}%**, "
        f"RankMe changes from **{_capped['rankme']['mean']:.2f} to {_uncapped['rankme']['mean']:.2f}**, "
        f"and balanced linear-probe accuracy changes by **{_probe_pp:+.1f} percentage points**. "
        "The paired intervals above show uncertainty over the five seeds. Read any smaller "
        "change-since-exit value alongside absolute reconstruction error; "
        "these measure different aspects of performance."
    )
    return


@app.cell
def _(Path, json, output, protocol):
    from cafl4ds.expert_lr_analysis import plot_baseline_curves

    baseline = Path(protocol["baseline_reference"]["directory"])
    baseline_comparison = json.loads((output / "baseline_comparison.json").read_text())
    return baseline, baseline_comparison, plot_baseline_curves


@app.cell
def _(baseline_comparison, mo):
    _keys = ("heldout_mse", "pass_2_end_minus_exit")
    _effects = baseline_comparison["expert_lr_minus_constant"]
    _includes_zero = all(
        _effects[_arm][_key]["ci95"][0] <= 0 <= _effects[_arm][_key]["ci95"][1]
        for _arm in ("learned", "gmm")
        for _key in _keys
    )
    mo.md(
        "**Original capped intervention versus M6.1.0:** the actual-update and displacement views below show whether "
        "low-mass experts are protected. "
        + (
            "The paired intervals for final MSE and second-pass deterioration include zero for both routers; "
            "these seeds do not establish an improvement over constant learning rates."
            if _includes_zero
            else "Inspect the paired baseline effects and individual seeds before drawing an overall conclusion."
        )
    )
    return


@app.cell
def _(baseline, baseline_comparison, mo, output, plot_baseline_curves):
    _rows = []
    for _arm in ("learned", "gmm"):
        for _metric in ("heldout_mse", "time_mean_mse_pass_1", "pass_2_end_minus_exit", "rankme"):
            _effect = baseline_comparison["expert_lr_minus_constant"][_arm][_metric]
            _conditions = baseline_comparison["conditions"][_arm]
            _rows.append(
                {
                    "Router": _arm,
                    "Metric": _metric,
                    "M6.1.0 constant LR": round(_conditions["constant"][_metric]["mean"], 6),
                    "M6.2.0 expert LR": round(_conditions["expert_lr"][_metric]["mean"], 6),
                    "Expert LR − constant": round(_effect["mean"], 6),
                    "Paired 95% interval": str([round(_v, 6) for _v in _effect["ci95"]])
                    if _effect["ci95"]
                    else "One seed",
                }
            )
    mo.vstack(
        [
            mo.md(
                "## Effect of expert learning rates versus M6.1.0\n"
                "Each row pairs the same router and seed across experiments. Initial weights, training IDs, "
                "order, image views and masks were checked at every step. Lower MSE and deterioration "
                "are better; higher RankMe means a higher effective representation rank. "
                "Time-mean MSE is the trapezoidal average over the recorded evaluations. "
                "Intervals are descriptive paired Student-t intervals over five training seeds."
            ),
            mo.ui.table(_rows, selection=None),
            plot_baseline_curves(output, baseline),
        ]
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
                "## Original capped arms: endpoints and paired differences\n"
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
        "**Capped-arm endpoint reconstruction:** GMM differs from learned routing "
        f"by **{_relative:+.2f}%** in mean MSE. "
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
    mo.vstack([mo.md("## Original capped arms: learning and standard health signals"), plot_learning(output)])
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
def _(health_metric, output, plot_conditions, uncapped_output):
    plot_conditions(uncapped_output, output, metric=health_metric.value)
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
def _(layer_control, mo, output, plot_three_rates, seed_control, uncapped_output):
    mo.vstack(
        [
            mo.md(
                "## Routing mass, applied learning rate and actual parameter movement\n"
                "The parameter-step row measures each expert's actual parameter-step L2 norm, including AdamW "
                "weight decay, on a shared logarithmic scale (zero updates are plotted at 1e-16). "
                "Rates use each minibatch's mass without "
                "additional smoothing. Expert IDs belong to the selected seed and layer."
            ),
            plot_three_rates(uncapped_output, output, seed_control.value, layer_control.value),
        ]
    )
    return


@app.cell
def _(json, layer_control, mo, output, seed_control, uncapped_output):
    _data = json.loads((output / "expert_displacement.json").read_text())
    _records = [
        _row for _row in _data["records"] if _row["seed"] == seed_control.value and _row["layer"] == layer_control.value
    ]
    mo.vstack(
        [
            mo.md(
                "### Expert displacement from the common initialization\n"
                "These are cumulative parameter distances at saved milestones, rather than sums of step norms. "
                "The table pairs the same expert across experiments; all saved initial tensors were checked exactly."
            ),
            mo.md("M6.1.0 constant rates → original M6.2.0 capped rates:"),
            mo.ui.table(_records, selection=None),
            mo.md("Capped GMM → added uncapped GMM:"),
            mo.ui.table(
                [
                    _row
                    for _row in json.loads((uncapped_output / "expert_displacement.json").read_text())["records"]
                    if _row["seed"] == seed_control.value and _row["layer"] == layer_control.value
                ],
                selection=None,
            ),
        ]
    )
    return


@app.cell
def _(mo, output, plot_added_regimes, regime_mode, seed_control, uncapped_output):
    mo.vstack(
        [
            mo.md(
                "## Per-regime reconstruction and retention\n"
                "All three arms share a color scale. Change-since-exit values restart each pass and are blank "
                "before that regime's block exits; positive values indicate worse reconstruction."
            ),
            plot_added_regimes(uncapped_output, output, seed_control.value, change=regime_mode.value == "change"),
        ]
    )
    return


@app.cell
def _(
    CONDITIONS,
    condition_folders,
    json,
    membership,
    mo,
    output,
    plt,
    read_rows,
    regime_control,
    seed_control,
    uncapped_output,
):
    _fig, _ax = plt.subplots(figsize=(13, 4), constrained_layout=True)
    for _condition, _folder in condition_folders(uncapped_output, output, seed_control.value).items():
        _rows = read_rows(_folder / "evaluation.jsonl")
        _ax.plot(
            [r["step"] for r in _rows],
            [r["regime_mse"][regime_control.value] for r in _rows],
            color=CONDITIONS[_condition][1],
            label=CONDITIONS[_condition][0],
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
                "## Original capped arms: which experts receive weight and gradients?\n"
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
                "These capped-arm values use exact per-image regime sums. "
                "The added uncapped GMM arm has identical routing, verified at every step."
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
