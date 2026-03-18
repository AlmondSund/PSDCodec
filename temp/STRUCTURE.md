# Project Structure

This document defines the intended repository contract for PSDCodec.

The tree below defines the intended organizational contract of the repository; some directories may remain partially populated until the corresponding implementation phases mature.

The goal is to keep theory, executable code, data assets, model artifacts, configurations, and generated outputs clearly separated so the repository remains navigable, reproducible, and easy to extend.

---

## Root Structure

```text
psdcodec/
├── docs/
│   ├── notes/
│   ├── articles/
│   ├── figs/
│   └── progress/
│
├── src/
│   ├── codec/
│   ├── data/
│   ├── models/
│   ├── objectives/
│   ├── pipelines/
│   ├── interfaces/
│   ├── infra/
│   └── utils/
│
├── data/
│   ├── raw/
│   ├── staged/
│   ├── schemas/
│   ├── metadata/
│   └── processed/
│
├── models/
│   ├── checkpoints/
│   ├── exports/
│   ├── benchmarks/
│   └── archive/
│
├── tests/
│   ├── specs/
│   ├── integration/
│   ├── e2e/
│   └── benchmarks/
│
├── scripts/
│   ├── jobs/
│   ├── tools/
│   ├── generate/
│   └── release/
│
├── configs/
│   ├── envs/
│   ├── docker/
│   ├── ci/
│   ├── runtime/
│   └── experiments/
│
└── reports/
    ├── figs/
    ├── benchmarks/
    ├── exports/
    └── archive/
```

---

## Artifact Flow

The repository is organized around artifact movement, not only artifact type.

- Human-authored explanation and manuscript material live in `docs/`.
- Executable logic lives in `src/`.
- Datasets move through `data/raw/` → `data/staged/` → `data/processed/`.
- Trained and exported model artifacts live in `models/`.
- Generated evaluation outputs and presentation-ready result bundles live in `reports/`.
- Reproducibility is supported by `configs/`, `scripts/`, and `tests/`.

This separation is deliberate: `docs/` explains the project, `src/` implements it, `data/` carries dataset lifecycle, `models/` stores produced model assets, and `reports/` stores produced evaluation assets.

---

## Folder Overview

### `docs/`

Contains human-authored project documentation.

- **`notes/`**: research notes, literature notes, exploratory writeups, and conceptual drafts.
- **`articles/`**: manuscript-oriented writing such as paper drafts, thesis chapters, and survey text.
- **`figs/`**: diagrams and documentation figures authored for explanation.
- **`progress/`**: human-written milestone notes, internal status updates, and progress summaries.

`docs/` is for material written by contributors. It is not the home for generated experiment outputs.

---

### `src/`

Contains executable source code only.

- **`codec/`**: codec-domain logic, including deterministic transforms, quantization, entropy-model-facing code, bitrate accounting, and reconstruction primitives. It must not contain training orchestration or service wiring.
- **`data/`**: dataset loading, validation, normalization, feature preparation, and data-pipeline utilities. It must not store datasets themselves.
- **`models/`**: executable learned model definitions, layers, and modules. It must not contain trained weights, serialized checkpoints, or declarative experiment settings.
- **`objectives/`**: losses, metrics, evaluation operators, and other mathematical criteria used for optimization or assessment. It must not contain pipeline control flow.
- **`pipelines/`**: training, evaluation, export, and experiment orchestration built by composing lower-level code. It must not become the home of reusable domain primitives.
- **`interfaces/`**: public-facing repository boundaries such as CLI entrypoints, API adapters, schema bindings, and external input/output contracts. It must not duplicate codec or model implementation.
- **`infra/`**: runtime support behind those boundaries, including logging, storage adapters, dependency wiring, hardware/runtime hooks, and backend-facing infrastructure. It must not define public interface contracts.
- **`utils/`**: small, dependency-light utilities used across modules. This directory is intentionally narrow and must not become a dumping ground for uncategorized logic.

`src/interfaces/` defines public-facing repository boundaries; `src/infra/` supports runtime execution behind those boundaries.

The purpose of `src/` is to keep domain logic, orchestration, interfaces, and infrastructure separate enough that each can evolve without absorbing the others.

---

### `data/`

Contains dataset assets and dataset-related structural material.

- **`raw/`**: original external datasets or untouched source acquisitions.
- **`staged/`**: normalized or imported data that has entered the project workflow but is still considered an intermediate form.
- **`schemas/`**: dataset format definitions, label schemas, and data contracts.
- **`metadata/`**: provenance, annotations, descriptive manifests, and dataset documentation.
- **`processed/`**: prepared datasets, derived subsets, and reusable processed outputs.

`data/` reflects lifecycle. New material should move forward through the pipeline rather than accumulating in ambiguous intermediate locations.

---

### `models/`

Contains produced model artifacts only.

- **`checkpoints/`**: training checkpoints and intermediate saved states.
- **`exports/`**: deployment-facing or exchange-ready model artifacts.
- **`benchmarks/`**: selected saved variants used for controlled comparison.
- **`archive/`**: deprecated, superseded, or retained historical model artifacts.

Executable model code belongs in `src/models/`. Declarative settings, including architecture variants and experiment parameters, belong in `configs/`, not here.

---

### `tests/`

Contains validation and quality-assurance code.

- **`specs/`**: unit and behavior tests for isolated components.
- **`integration/`**: tests for interactions across subsystems.
- **`e2e/`**: end-to-end workflow validation.
- **`benchmarks/`**: repeatability, regression, and performance checks.

`tests/benchmarks/` contains executable benchmark and regression checks. Generated benchmark tables, plots, and summaries belong in `reports/benchmarks/`.

Tests should validate behavior and invariants, not merely mirror directory names.

---

### `scripts/`

Contains executable operational scripts.

- **`jobs/`**: named training, evaluation, inference, or experiment runs.
- **`tools/`**: maintenance and inspection utilities.
- **`generate/`**: controlled generators for templates, scaffolds, manifests, or other derived project files.
- **`release/`**: packaging, export, and release-preparation scripts.

Scripts may call into `src/`, but they should not reimplement source logic that ought to live in reusable modules.

---

### `configs/`

Contains declared configuration and execution settings.

- **`envs/`**: environment definitions for local, server, or lab execution.
- **`docker/`**: container and image configuration.
- **`ci/`**: continuous integration and automation definitions.
- **`runtime/`**: runtime, service, and deployment-facing settings.
- **`experiments/`**: training, evaluation, export, and ablation parameters, including declarative model-variant choices.

If a setting can be declared rather than hard-coded, it should generally live here.

---

### `reports/`

Contains generated outputs and result artifacts.

- **`figs/`**: generated plots, charts, and visual experiment outputs.
- **`benchmarks/`**: benchmark tables, comparison summaries, and evaluation outputs.
- **`exports/`**: publication-ready or presentation-ready generated bundles.
- **`archive/`**: preserved historical outputs and superseded result sets.

`reports/` is for produced outputs, not for narrative project updates. Human-written progress material belongs in `docs/progress/`.

---

## Structural Rules

This structure is governed by a few simple rules.

- **Documentation and outputs are different artifact classes.** Human-authored explanation belongs in `docs/`; generated outputs belong in `reports/`.
- **Executable code and produced artifacts are different artifact classes.** Source code belongs in `src/`; trained checkpoints and exported models belong in `models/`.
- **Configuration is explicit.** Runtime settings, experiment parameters, and declarative architecture choices belong in `configs/` rather than being scattered across code and artifact folders.
- **Orchestration is not domain logic.** `src/pipelines/` may compose lower-level modules, but codec math and reusable primitives belong in `src/codec/`, `src/models/`, `src/data/`, or `src/objectives/`.
- **Interfaces and infrastructure are not interchangeable.** Public contracts belong in `src/interfaces/`; runtime support behind those contracts belongs in `src/infra/`.
- **Utilities stay narrow.** `src/utils/` is reserved for small shared helpers; if code has a clear domain meaning, it should live in a domain directory instead.
- **Lifecycle should be visible.** Data and artifacts should move through explicit stages rather than accumulate in unclear mixed-use directories.

---

## Placement Guide

When placement is unclear, use the following fallback rules.

- put **human-written documentation** in `docs/`
- put **executable implementation** in `src/`
- put **dataset assets and dataset contracts** in `data/`
- put **trained or exported model artifacts** in `models/`
- put **tests and benchmark checks** in `tests/`
- put **operational scripts** in `scripts/`
- put **declared settings and environment definitions** in `configs/`
- put **generated figures, tables, and result bundles** in `reports/`

When in doubt, choose the directory that best preserves traceability of origin and role.

---

## Goal

The purpose of this structure is to keep repository boundaries durable and artifact lineage traceable as theory, implementation, experiments, and documentation evolve together.
