"""M6.3.0 reconstruction and forgetting across the day+dusk to night switch."""

# mypy: ignore-errors

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os
    from pathlib import Path

    import marimo as mo

    from cafl4ds.day_night_analysis import LABELS, plot_curves, read_comparison, run_directory

    return LABELS, Path, json, mo, os, plot_curves, read_comparison, run_directory


@app.cell
def _(Path, json, mo, os, read_comparison):
    output = Path(os.environ.get("M630_ARTIFACT_DIR", Path(__file__).resolve().parents[1] / "outputs/m6_3_0"))
    mo.stop(not (output / "comparison.json").exists(), mo.md("The full comparison is still running."))
    protocol = json.loads((output / "protocol.json").read_text())
    comparison = read_comparison(output)
    return comparison, output, protocol


@app.cell
def _(comparison, mo, protocol):
    _run = comparison["runs"][0]
    _counts = _run["images_per_phase"]
    _val = protocol["validation_counts"]
    _architecture_count = len(comparison["conditions"])
    _ablation_note = (
        "The four-layer GMM MoE uses shared attention, GMM-weighted MLP experts in every block, and a shared decoder."
        if "gmm_all_layers" in comparison["conditions"]
        else ""
    )
    mo.md(
        f"""
        # M6.3.0 · Day+dusk to night forgetting

        **{len(protocol["config"]["seeds"])} paired seeds, {_architecture_count} architectures,
        one pass per time partition.**

        Each run trains on {_counts["day_dusk"]:,} day+dusk images, then
        {_counts["night"]:,} night images. The switch follows update {_run["boundary_step"]:,}.
        Validation uses a fixed official BDD100K panel with {_val["day"]} daytime,
        {_val["dusk"]} dawn/dusk and {_val["night"]} night images.
        Each image contributes the average MSE across {protocol["config"]["mask_draws"]} fixed views and masks.

        Forgetting is the final day+dusk masked-patch MSE minus MSE measured immediately
        before the first night update. Positive values indicate an increase in error.
        The night curve shows second-phase learning and the absolute day+dusk curve
        shows how well each model learned before the switch.

        The learned gates route at image level. GMM arms use fixed image posteriors and
        capped private learning-rate multipliers of `min(1, 4 × mean batch probability)`.
        Private encoder trunks mix their final tokens before decoding except for the
        private-decoder arm, which mixes the final reconstructed patches.
        {_ablation_note}
        """
    )
    return


@app.cell
def _(LABELS, comparison, mo):
    _means = []
    for _arm in comparison["conditions"]:
        _stats = comparison["conditions"][_arm]
        _run = next(_record for _record in comparison["runs"] if _record["arm"] == _arm)
        _means.append(
            {
                "Arm": LABELS[_arm],
                "Day+dusk at switch": round(_stats["day_dusk_pre_night"]["mean"], 5),
                "Day+dusk final": round(_stats["day_dusk_final"]["mean"], 5),
                "Forgetting Δ MSE": round(_stats["forgetting_mse"]["mean"], 5),
                "Peak Δ MSE": round(_stats["peak_forgetting_mse"]["mean"], 5),
                "Night final": round(_stats["night_final"]["mean"], 5),
                "Parameters": _run["parameters"],
                "Forward GFLOPs/image": round(_run["approx_forward_gflops_per_image"], 4),
            }
        )
    mo.vstack(
        [
            mo.md(
                "## Seed means\n"
                "Parameter counts are exact. Forward GFLOPs estimate the executed matrix multiplies in masked MAE "
                "training, with two FLOPs per multiply-add. The training budget matches image exposure."
            ),
            mo.ui.table(_means, selection=None),
        ]
    )
    return


@app.cell
def _(LABELS, comparison, mo):
    _effects = []
    for _effect in comparison["paired_forgetting_differences"]:
        _ci = _effect["ci95"]
        _effects.append(
            {
                "Arm": LABELS[_effect["arm"]],
                "Reference": LABELS[_effect["reference"]],
                "Paired Δ forgetting": round(_effect["mean"], 5),
                "Descriptive 95% interval": f"[{_ci[0]:+.5f}, {_ci[1]:+.5f}]" if _ci else "One seed",
            }
        )
    mo.vstack(
        [
            mo.md(
                "## Paired differences in forgetting\nNegative differences favor the named arm. Intervals use seeds."
            ),
            mo.ui.table(_effects, selection=None),
        ]
    )
    return


@app.cell
def _(mo, protocol):
    _subsets = {"Day + dusk": "day_dusk", "Daytime": "day", "Dawn/dusk": "dusk"}
    subset_control = mo.ui.dropdown(_subsets, value="Day + dusk", label="Validation subset")
    _options = {"Mean across seeds": None, **{f"Seed {_seed}": _seed for _seed in protocol["config"]["seeds"]}}
    seed_control = mo.ui.dropdown(_options, value="Mean across seeds", label="Seed")
    mo.hstack([subset_control, seed_control], justify="start")
    return seed_control, subset_control


@app.cell
def _(output, plot_curves, seed_control, subset_control):
    plot_curves(output, subset_control.value, seed_control.value)
    return


@app.cell(hide_code=True)
def _(comparison, mo, output, run_directory, seed_control, subset_control):
    from cafl4ds.day_night_drift import plot_representation_drift

    _folders = [
        run_directory(output, run)
        for run in comparison["runs"]
        if seed_control.value is None or run["seed"] == seed_control.value
    ]
    mo.stop(
        not any(
            all((folder / filename).is_file() for folder in _folders)
            for filename in ("drift.json", "drift_checkpoints.json")
        ),
        mo.md("Representation metrics are unavailable for this selection."),
    )

    mo.vstack(
        [
            mo.md(
                "## Representation drift\n"
                "Linear CKA and cosine drift use unmasked encoder embeddings for the same validation images "
                "at each step. Each seed is compared with its own initialization and pre-night checkpoint."
            ),
            plot_representation_drift(output, subset_control.value, seed_control.value),
        ]
    )
    return


@app.cell
def _(LABELS, comparison, mo):
    _rows = [
        {
            "Arm": LABELS[_run["arm"]],
            "Seed": _run["seed"],
            "Day+dusk at switch": round(_run["pre_night"]["day_dusk_mse"], 6),
            "Day+dusk final": round(_run["final"]["day_dusk_mse"], 6),
            "Forgetting Δ MSE": round(_run["forgetting_mse"], 6),
            "Night final": round(_run["final"]["night_mse"], 6),
        }
        for _run in comparison["runs"]
    ]
    mo.vstack([mo.md("## Individual seeds"), mo.ui.table(_rows, selection=None)])
    return


if __name__ == "__main__":
    app.run()
