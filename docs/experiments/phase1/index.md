# Phase 1 — Degradation Envelope `[NEW]`

This is the landing page for Phase 1: it states the goal, the methodology, the exit gate, and indexes the studies the
phase is composed of. Detail lives one level down, in the per-study docs linked from the [Studies](#studies) table.

## Goal (from the [project plan](../../project-plan/index.md#plan))

The relevant part of the project plan architecture is the following open loop, the complement of the project's
[closed loop](../../project-plan/index.md#scope). The forward pipeline (data → filter → backbone → monitor) and the
intrinsic **fast** edge (the filter scores against the current backbone) are live; the **slow** edge (monitor re-aims
selection) is left **open** — health is read out and logged, not fed back. Closing that edge is Phase 3's job.

```
data stream → [selection filter] → SSL backbone (adapts online) → [health monitor]
                   ▲   ▲                                                │
     fast edge ────┘   └ ─ ─ ─ ─ ─ ─ slow edge (monitor → filter) ─ ─ ─ ┘
   (model→filter)
```

**Why this phase is the hinge.** The project's contribution is a *live loop* — a selection filter feeds a
self-supervised backbone, and a health monitor watches the backbone and re-aims the filter — under one governing
discipline: *you cannot close a loop to correct a pathology you have not first shown exists*. That discipline is what
sequences the whole plan. Phase 0 **calibrates** the health instruments, but only on *deliberately-broken* models at toy
scale; Phase 1 **demonstrates the phenomenon** those instruments were built for; Phase 2 studies which selection
criterion wins under the resulting dynamics; Phase 3 finally **closes the loop**. So Phase 1 is the make-or-break pivot,
and everything downstream is conditional on it. The distinction that matters: calibration proved only that a lit match
trips the smoke detector; Phase 1 asks whether the house actually catches fire when a backbone lives on a real,
correlated stream. That makes this a question of **existence, not measurement**.

The concrete job is to **map the degradation envelope of the coupled streaming loop** — to show that running a
self-supervised backbone on a correlated, single-pass stream actually does something measurable to its representation,
and that the pieces we can turn (adapt vs. freeze, which frames we select) have leverage over that outcome. Concretely,
three existential legs, all read against the Phase-0 gates:

- **(a) Adaptation beats frozen.** An online-adapting backbone must beat the frozen-backbone floor (**B5**). If it does
    not, there is nothing to protect.
- **(b) Streaming induces measurable degradation.** A correlated diet must move the health trajectory in a way the
    calibrated instruments catch — and the per-mode positive control (**PC**) must still fire.
- **(c) The selection knob moves health.** Swapping a cheap selection filter must measurably shift the trajectory —
    evidence that the fast (model→filter) edge is a real lever, not a passive readout.

The outcome is a gate based on a `pass` result for each of the gates above:

- **Go** — a coupling exists on real data → proceed to Phase 2 (open-loop criterion study).
- **No-Go** — no coupling survives stress → reframe the project to *selection-for-efficiency* and say so plainly.

## Executive Summary

*Placeholder — written once the runs land. It will read as the Phase-0 index's does: the honest verdict on whether the
coupling exists, which legs passed on which data/init, and the load-bearing caveats Phase 2 must carry.*

## Methodology

**Open loop only.** Phase 1 exercises the *fast* edge (the filter scores informativeness against the continuously
adapting backbone) and *builds* the health monitor, but it does **not** close the *slow* edge (monitor→filter re-aiming)
— that is Phase 3. So the independent variable here is the **base filter**, fixed per run, using stock knobs only
(no-filter / reservoir / dedup / loss). We are not yet designing a filter; we are asking whether the system this loop
creates has pathologies worth correcting. See [novelty.md](../../project-plan/novelty.md) (N-A) for the coupling claim
this phase underwrites.

**Two modes, prioritized — not symmetric.** Phase 1 reads degradation on the two backbones Phase 0 calibrated, but which
one and how much effort each gets is *given* by Phase 0, not chosen for balance. **Forgetting on MAE is the primary
target** — it is the project's core continual-learning worry and Phase 0's marquee open question (a from-scratch MAE
*resists* forgetting; whether correlation × a long horizon finally breaks that resistance is unresolved). Prior work
sharpens this target rather than softening it: continual-MAE *pretraining* independently finds MAE forgets little and
that replay trivially protects it (the continual-MAE line — see
[reading list](../../project-plan/reading-list.md#position-against-work-we-must-distinguish-ourselves-from)), always
under random / replay sampling with no filter in the loop. So the novel question is not *whether* MAE forgets — prior
art suggests: not easily — but whether **selection** moves the trajectory, and how far the envelope must be pushed. P1.3
is scoped to that reframe (map the envelope, test the selection lever), not to hunting a spontaneous crater. **Collapse
on a joint-embedding backbone is secondary** — already well-calibrated, so the job is to confirm it *shows up* under the
diet, not to resolve an open question. The two are near-different experiments — different backbone, different reader
(projector geometry vs. drift/BWT), different substrate ladder (see [Goal](#goal)), different Phase-0 status — which is
why each gets its **own** study (P1.3 forgetting, P1.4 collapse). The other two calibrated modes are **carried, not
primary vehicles**: instability is a training-recipe *choice* (below), and MAE representation-*quality* has no
label-free internal reader, so Phase 1 only *provisions* for it (a labelled canary or a JE projector) rather than
chasing it.

**Read movement against the gate, never in isolation.** Every run logs the positive control and the frozen floor B5
beside the live arms, because a Phase-0 lesson is load-bearing here: **a signal is only interpretable against its
gate**. This is Phase 1's hardest new problem — the loop's raw instrument *movement* cannot, by itself, tell genuine
degradation from benign reorganization (a healthy backbone reorganizing has, e.g., a U-shaped RankMe dip;
[P0.1.0](../phase0/P0.1.0.md), [P0.2.1](../phase0/P0.2.1.md)). Producing that discrimination — a healthy-movement null
plus a protocol for reading movement against it — is a **named deliverable (P1.1)**, not an assumption the mode studies
get to make.

**Initialization is a factor, not a fixed choice.** A from-scratch backbone is the degradation-sensitive setting and the
natural home for the demonstration; a fully-pretrained one sits in a good basin, adapts gently, and can **mask**
collapse into a false negative. The two carry different senses of "degradation" — *failing to form* good features
(collapse) vs. *eroding* ones it already had (forgetting) — and both are worth eliciting, so init is swept, never
locked.

**What Phase 1 carries in from Phase 0.** The [handover to Phase 1](../phase0/index.md#handover-to-phase-1) is the
operative to-do list; the load-bearing items:

- *Collapse* — anchor the health read to a **fixed, independently-certified healthy reference**, not a co-trained
    "healthy" arm that would drift with the very signal being read; the *principle* ports even though Phase 0's toy
    anchor and its firing numbers do not. Expect the target corner (correlated × single-pass) to **thin the collapse
    suite** — possibly to a single marginal reader — so budget a second reader and re-establish the bar at scale
    ([P0.2](../latest-results/P0.2.md)).
- *Forgetting (MAE)* — MAE **resists**: severity is manufacturable but every toy crater stayed recoverable, and no cheap
    lever reached the *permanent* pole ([P0.3](../latest-results/P0.3.md), P0.3.9/P0.3.10). Prior work agrees —
    continual-MAE *pretraining* finds the same resistance, and that replay trivially protects it (the continual-MAE
    line, [reading list](../../project-plan/reading-list.md#position-against-work-we-must-distinguish-ourselves-from)) —
    so the open question is not *whether* it forgets but whether **selection** and a hard-enough envelope (correlation ×
    long horizon) move it. Two carried cautions: the **LR schedule** is itself a forgetting confound (re-warming
    manufactures it independently of the diet; Beyond Cosine Decay, CoLLAs 2025), so pin and sweep it; and label-free
    **probing can understate** MAE forgetting (fine-tuning reflects it better), so corroborate with a fine-tune probe,
    not kNN / linear alone.
- *Forgetting (joint-embedding)* — read it via representation **drift**, not the reused collapse geometry
    ([P0.6](../latest-results/P0.6.md)).
- *Instability* — the grad-norm early warning is optimizer/loss-contingent (raw MSE + SGD); pick the training recipe
    with that tradeoff in view ([P0.4](../latest-results/P0.4.md)).
- *Representation quality (MAE)* — is **not** readable label-free from MAE internals; provision a labelled canary or a
    joint-embedding projector ([P0.5](../latest-results/P0.5.md)).

**The substrate ladder, and why STL-10 is only a smoke-test.** STL-10 earns its place solely as the engineering
smoke-test (P1.0): Phase 0 showed it can host collapse but *cannot* induce forgetting even under a deliberate control,
so reading dynamics off it would be a class-blocking artifact at best and impossible at worst. Prior art adds a second
reason: the **tunnel effect** is *amplified* by few-class, low-resolution data (Masarczyk 2023), so a rank drop on
STL-10 is especially likely to be benign compression rather than genuine degradation. The scientific claim rests on
BDD's real correlation; the mid rung between them exists to **de-risk BDD** — so that a null there reads as "no
coupling" rather than "underpowered / mis-configured" (see [risks.md](../../project-plan/risks.md),
synthetic-correlation artifact).

## Exit Gate

Phase 1 exits when the degradation envelope has been mapped **on real data (BDD100K)** and the coupling is resolved
either way:

- **Go** requires all three legs on real BDD correlation — (a) adaptation beats B5, (b) streaming induces
    instrument-caught degradation with the PC firing, (c) the selection knob moves the health trajectory — demonstrated
    in at least the primary vehicle (forgetting) or, failing that, cleanly in the secondary (collapse); with the loop's
    apparatus smoke-test passed beforehand, the degradation **corroborated downstream** (a real probe / kNN / few-shot
    drop, not just a moved needle), and the init-dependence of the effect characterized. We look hardest at forgetting
    because that is Phase 0's open question, but a coupling that manifests as collapse still counts.
- **No-Go** is a legitimate exit: the coupling is shown absent under stress, and the project reframes to
    selection-for-efficiency (the reframe is recorded, not buried).

## Studies

The ordering below follows the scientific dependency: build the loop ([P1.0](../latest-results/P1.0.md)), earn the right
to *read* it (P1.1) and the right to *continue* (P1.2, the existential floor), then run the two prioritized
mode-dynamics studies up their substrate ladders to BDD (P1.3 forgetting, primary; P1.4 collapse, secondary), with a
cross-cutting regime-and-downstream check (P1.5). The FL and monitoring tracks (P1.6) run in parallel from the start.
The three existential legs map to studies as: **(a) → P1.2**; **(b) and (c) → read off the *same* runs inside P1.3 /
P1.4** (you cannot vary the knob without also reading degradation), landing on BDD via each study's top rung.

Studies and their sub-studies share **one table**. A study appears as a single `P1.x` row until its work begins; once it
is broken down, that row is **replaced** by its `P1.x.y` sub-study rows (so nothing is stated at both levels). Sub-study
rows follow the Phase-0 conventions (the "What it establishes" cell opens with `Motivation:`; the Status cell is
`<marker> **<label>** (<finding>)`), and their detail lives in `phase1/<id>.md`, with the concise take-away in
`latest-results/<id>.md`. So far only **P1.0** is broken down (into the `P1.0.0` / `P1.0.1` / `P1.0.2` rows below); the
rest still show as single study rows.

| ID | Study / sub-study | What it establishes | Status |
| -- | -- | -- | -- |
| P1.0.0 | Loop-integration harness `[engineering]` | Motivation: nothing else in Phase 1 runs without one *working open loop*. Wire the stream, the selection filter, the SSL backbone, and the health monitor into one loop, consolidating the scattered Phase-0 pieces — B5 measured only at run-end ([run_loop.py](../../../scripts/run_loop.py) `adaptation_report`), a separate positive-control script per mode each emitting its own `comparison.json` — into **one multi-arm harness** that co-logs the **live / PC / B5** arms into a single **canonical `comparison.json`** (the base schema is owned here; later studies extend it but can always fall back to it). This goes beyond P0.1.0's ~30-step harness ([P0.1.0](../phase0/P0.1.0.md); stream + backbone + logged instruments) by adding the filter and the monitor as *components* and PC/B5 as loop-integrated baselines; it assembles the known streaming-SSL harness (Purushwalkam's continuous-SSL setup 2022; Memory Storyboard 2025) — the novelty is the coupling, not the substrate. Default to Phase-0's proven **sequential same-seed arms** (each a `StreamingLoop.run()`, reused untouched; B5 = a frozen, measured-only arm), aligned post-hoc on the shared step index. Prove it on the full 2×2 **C×I** STL-10 smoke — the rig runs, instruments move in the right *direction*, the PC fires, nothing NaNs, **no degradation numbers** — plus a **long-horizon** resampled-image run that stays finite with flat memory (the no-blow-up / no-leak check). | ✅ **Built** (full C×I STL-10 smoke green — rig runs, live reproduces P0.1.0, PC fires, B5 frozen, finite + flat-memory) |
| P1.0.1 | Arm-alignment divergence (same-seed vs shared-iterator) `[analysis]` | Motivation: the sequential same-seed arms rest on an unverified assumption — that re-seeding each arm reconstructs a **bit-identical** stream and keeps the arms' trajectories aligned for the post-hoc `zip` (the alignment method every Phase-0 PC already inherits). Test it against a **shared-iterator co-stepped** harness (the oracle — provably identical data by construction): measure (1) per-step **data identity** and (2) **health-trajectory divergence**, attributing any drift to stream *construction* vs. *in-arm* RNG (augmentation, MAE masking, dropout, BatchNorm). Retroactively certifies the Phase-0 comparison method; the co-stepped harness is retained as an optional oracle mode. Owns a number (divergence magnitude) — which is *why* it is a sub-study, not folded into the number-free P1.0.0. | ✅ **Complete** (same-seed `zip` certified — rebuild bit-identical, matched trajectory zero, in-arm-RNG control sensitive; oracle retained as optional mode) |
| P1.0.2 | Deployment-prototype de-risker harness `[engineering]` | Motivation: the smoke (P1.0.0) and its certificate (P1.0.1) prove the loop *runs* on toy data, but nothing yet de-risks the **real substrate** the exit gate stands on — STL-10 is a smoke-test by construction (can't induce forgetting, amplifies benign tunnel-effect rank drop), so a BDD null is only interpretable once the ingestion + diet + logging are proven to run and read sensibly. Build a **deployment prototype** that takes the certified open loop onto **BDD100K** (images, not video) with a *meaningful, configurable* learning diet (default: correlated `(timeofday, weather)` era blocks walking a `day·clear → day·rain → dusk → night` shift, long-tailed, single-pass — the real-attribute analogue of the synthetic `class_blocked` knob), a **config-selected backbone** (`backbone=mae\|je`, one per run — same seed rebuilds the same regime by P1.0.1's certificate, so no precarious co-step), **optional** PC/B5 gate arms, and **Phase-0-informed multivariate logging**: the full instrument vector each checkpoint, label-free *and* labelled-canary channels (labels allowed — no deployment constraint yet), every signal tagged with its Phase-0 calibrated status via a code trust-map registry stamped into the artifact. Compute-agnostic core (one `device` knob spans cpu/cuda/hpu; `img_size` downscales to fit memory), hardware only at the existing edges (`docker/gaudi.env.Dockerfile`, `run_gaudi_dev.sh`). Reads **no degradation** — a two-tier validation (anyone-can-run wiring check + our sanity-of-readings pass); the diet's scientific adequacy and every degradation read stay P1.1 / P1.3 / P1.4. Primary deliverable = the **config + promoted logged-signal corpus**; the pipeline is the means. New entry point `scripts/run_deploy_harness.py` + `deploy.yaml`. | ✅ **Complete — both tiers green on real BDD100K** (BDD substrate + configurable regime diet + config-selected backbone + warm-well step + seed-ensemble drive + trust-annotated multivariate corpus with a label-tagged leg map; Tier-A green on the synthetic wiring check — no dataset or accelerator, the whole loop runs on a plain CPU in seconds — Tier-B sanity-of-readings green on the BDD100K val split on a Gaudi2 HPU (for throughput — the drive runs on CPU too, just slower) — three promoted 5-seed ensembles (warm/cold JE, warm MAE) over the 17-leg walk across BDD's 18 driving regimes, all finite, RankMe in range, drift accumulating, scene-canary above chance) |
| P1.1 | Reading protocol — degradation vs. benign reorganization | Motivation: leg (b) is uninterpretable without it. Phase 0's load-bearing warning is that a healthy backbone reorganizing *also* moves the needles (a U-shaped RankMe dip, rising drift) for benign reasons ([P0.1.0](../phase0/P0.1.0.md), [P0.2.1](../phase0/P0.2.1.md)). Establish the **healthy-movement null** on an IID control stream and the protocol for reading each instrument's *movement against its calibrated gate*, so a dip or a climb can be called degradation rather than reorganization. Two load-bearing priors sharpen the protocol: benign drift is **non-mean-reverting change without performance loss** while forgetting loses information (discriminate via mean-reversion + probe-recovery), and a **numerical-rank drop in deep layers is benign tunnel-effect compression** for the trained task (it hurts only OOD, and is *amplified* by toy low-res / few-class data — another reason STL-10 is smoke-test only; Masarczyk 2023). Ports the durable Phase-0 anchoring *principle* — a fixed, independently-certified reference, not a co-trained arm — and the matched-class era-eval fix. Adopt **CKA** (checkpoint-to-reference representation similarity, Kornblith 2019) here as a *provisional* label-free lens: it localizes *where in the network* drift happens — a discrimination the scalar instruments lack — but carry its reliability caveat (CKA is sensitive to outliers / function-preserving shifts; use the debiased estimator + a complementary metric like Procrustes/CCA — Davari 2023); it rides this study's healthy-movement null rather than a Phase-0 re-open, staying an analysis lens (not a Go/No-Go gate) until it earns promotion. Likewise calibrate **intrinsic dimension** here as a candidate label-free instrument Phase 0's suite did not include — a 260-model comparison finds it the most reliable single label-free downstream proxy (Arputharaj et al., TMLR 2026; estimator: IdEst, ICML 2026), but its reliability too is moderated by architecture/objective, so it enters as a *suite member* held to the same healthy-movement null (fire on degradation, stay quiet on benign reorganization), not a standalone gate — and calibrated **here, not via a Phase-0 re-open** (the same disposition as CKA). It then feeds P1.5's downstream-corroboration panel. Delivers the discrimination P1.3 / P1.4 then rely on. | 🔲 **Not started** |
| P1.2 | Existential floor — does adaptation beat frozen? (leg a) | Motivation: the project assumes online adaptation beats freezing the backbone; if it does not (B5 ≥ adapted — the RanDumb worry), Phase 1 halts and the project reframes to selection-for-efficiency. Prior art makes this a **genuinely contested** leg, not a formality: RanDumb (NeurIPS 2024) shows random features **outperform** online continually-learned representations in exactly this single-pass regime — but for *discriminative* CL, leaving the SSL case open — while Memory Storyboard (2025) shows streaming-SSL adaptation *can* beat frozen. Test adapted-vs-B5 (frozen-random / frozen-pretrained, init-matched) on the ascending-label evals (kNN → linear probe → few-shot), across the init sweep. Runs early and cheap because a null here is fatal. | 🔲 **Not started** |
| P1.3 | Forgetting dynamics (primary) — does correlation × horizon break MAE's resistance? | Motivation: the primary Phase-1 vehicle and Phase 0's marquee open question. A from-scratch MAE *resists* forgetting — severity is manufacturable but every toy crater stayed recoverable, and no cheap lever reached the *permanent* pole ([P0.3.9](../phase0/P0.3.9.md)/[P0.3.10](../phase0/P0.3.10.md)); the standing hypothesis is that correlation × a long horizon at real scale breaks it. Prior art reframes the target: continual-MAE *pretraining* already shows MAE resists and replay trivially protects it ([reading list](../../project-plan/reading-list.md#position-against-work-we-must-distinguish-ourselves-from)), so the defensible, novel finding is the **envelope** (how hard you must push before resistance breaks) and the **selection lever** (leg c) — not a spontaneous crater prior work says is unlikely. The streaming-SSL degradation line (Purushwalkam's MinRed 2022; Memory Storyboard 2025) is the closest prior to the *premise*: temporal correlation degrades streaming SSL and de-correlating replay mitigates — our distinction is steering selection by health, live. Pin and sweep the **LR schedule** as a confound (re-warming manufactures forgetting independently of the diet), and corroborate with a **fine-tune** probe (label-free probing can understate MAE forgetting). Run the forgetting battery up its substrate ladder — a mid rung where forgetting can be induced (STL-10 could not, [P0.3](../latest-results/P0.3.md)) → BDD — pushing the pressure sweep toward the permanent pole (savings-to-criterion at real horizon). Read **legs (b)+(c) off the same runs**: does the diet degrade (b), and does swapping the stock knob (B-floor / reservoir / dedup / loss) move that degradation (c). Contrast the joint-embedding vehicle, where forgetting turns on under a far shift and is drift-readable ([P0.6](../latest-results/P0.6.md)). Owns the exit-gate (b)+(c) claim on real data. | 🔲 **Not started** |
| P1.4 | Collapse dynamics (secondary) — does the diet collapse a joint-embedding backbone? | Motivation: the secondary vehicle — collapse is already well-calibrated, so the job is to confirm it *shows up* under the streaming diet, not to resolve an open question. The mechanism is **predicted but not demonstrated**: Jing (2022) shows dimensional collapse strikes directions where augmentation variance exceeds *data* variance, so a low-diversity / correlated diet is *more susceptible* — but no prior work shows collapse from a correlated *stream*, live, coupled to selection (DiSF selects diverse LLM-pretraining files offline). **Control two confounds** that cause collapse independent of the diet — streaming's small effective batch and a small from-scratch model (SimSiam is size-sensitive) — sweeping both with model size (P1.5). On a joint-embedding backbone (read at the projector), run the collapse battery STL-10 → BDD, reading **legs (b)+(c) off the same runs** (degradation + knob-leverage). Carry the Phase-0 caveat: in the target corner (correlated × single-pass) the collapse suite thins, possibly to a single marginal reader, so a second reader is budgeted (P1.1 anchoring; [P0.2](../latest-results/P0.2.md)). Corroborates the coupling; a collapse null does not by itself sink the phase if forgetting clears. | 🟡 **In progress** (P1.4.0 STL-10 complete; BDD pending) |
| P1.5 | Initialization × pressure regime map + downstream corroboration | Motivation: two cross-cutting checks the exit gate needs. First, the coupling may be **regime-specific** — from-scratch is degradation-sensitive, while a fully-pretrained backbone sits in a good basin and can *mask* collapse into a false negative; sweep init {from-scratch, lightly-pretrained, fully-pretrained} against the pressure knobs (correlation, LR, model size, horizon, replay on/off) to locate onset and report the init-dependence *as* a finding (failing to form features vs. eroding them). The init/depth dependence is grounded in the **tunnel effect** (Masarczyk 2023): overparameterization + few classes + low resolution lengthen the compressing "tunnel", so from-scratch toy regimes both degrade OOD more and amplify benign rank drop — reinforcing why init is swept and STL-10 is smoke-test only. Second, **downstream corroboration** — confirm the health-metric degradation tracks a real downstream drop (probe / kNN / few-shot), so we have moved the patient, not just the thermometer ([risks.md](../../project-plan/risks.md)). Read a metric **suite**, not one needle: RankMe is necessary-but-not-sufficient (Garrido 2022), and a 260-model comparison (Arputharaj et al., TMLR 2026) finds **intrinsic dimension** the most reliable single label-free downstream proxy — with a concrete estimator in IdEst (ICML 2026) — a candidate corroboration signal, though its reliability too is moderated by architecture/objective. For MAE specifically, weight the **fine-tune** probe: label-free probing can understate MAE forgetting (fine-tuning reflects it better), so kNN / linear alone may read false health. Cross-cuts P1.3 / P1.4 rather than standing alone. | 🔲 **Not started** |
| P1.6 | Parallel tracks — FL infrastructure + health-monitor seed | Motivation: two tracks the plan starts in parallel from Phase 1. The FL infrastructure (FedAvg + client simulation) is built now so the FL science can slot in right after the Phase-2 centralized reference; the health monitor is seeded once Phase 1 confirms degradation exists (its required input) so it is ready for the Phase-3 closed loop. Enablement, not a Go/No-Go leg. | 🔲 **Not started** |

The first collapse sub-study is detailed in
[P1.4.0](P1.4.0.md): intact SimSiam under correlated STL-10 training, with matched training
budgets, downstream corroboration, and a small selection/replay pilot. Its local-CPU core study
is complete: the 128-image class-block diet changed projector geometry and produced a smaller
quality penalty, while both selection interventions improved stressed quality below the registered
3-point protection margin ([results](../latest-results/P1.4.0.md)). Single-pass and BDD transfer,
along with the wider P1.4 scope, remain pending.

## Artifacts

*Conventions inherited from Phase 0, to be populated as runs land.* Each study's harness run is expected to write a
`comparison.json`-style artifact (gate verdict + per-`(instrument × surface)` series for the live / PC / B5 arms) into
its gitignored Hydra run dir; the runs that back a study's cited numbers are then promoted, tracked and slimmed, under
`artifacts/<substudy-id>/`, with the per-substudy listing living in each study's own detail doc. The precise per-harness
schema is fixed per study once its harness exists — see the Phase-0 [Artifacts](../phase0/index.md#artifacts) section
for the policy this phase follows.
