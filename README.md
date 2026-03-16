# PSDCodec
PSDCodec: a neural codec for power spectral density compression

# Project Structure

This repository is organized to support both:

- **professional software / AI model development**
- **research, thesis, and paper production**

The folder names are also chosen to work well with the **Material Icon Theme** extension in VS Code, so the project stays both structured and visually clear.

---

## Root Structure

```text
psdcodec/
├── docs/
│   ├── notes/
│   ├── articles/
│   ├── figs/
│   └── reports/
│
├── src/
│   ├── core/
│   ├── data/
│   ├── models/
│   ├── funcs/
│   ├── services/
│   ├── api/
│   ├── backend/
│   └── helpers/
│
├── data/
│   ├── downloads/
│   ├── imports/
│   ├── schemas/
│   ├── metadata/
│   └── exports/
│
├── models/
│   ├── schemas/
│   ├── exports/
│   ├── benchmarks/
│   └── archive/
│
├── tests/
│   ├── specs/
│   ├── integration-tests/
│   ├── e2e/
│   └── benchmarks/
│
├── scripts/
│   ├── jobs/
│   ├── tools/
│   ├── gens/
│   └── release/
│
├── configs/
│   ├── envs/
│   ├── docker/
│   ├── ci/
│   └── settings/
│
└── reports/
    ├── figs/
    ├── benchmarks/
    ├── exports/
    └── archive/
````

---

## Folder Overview

### `docs/`

Contains all human-readable project documentation.

* **`notes/`**: research notes, literature notes, ideas, summaries, and conceptual drafts.
* **`articles/`**: thesis chapters, paper drafts, survey text, and manuscript-oriented writing.
* **`figs/`**: diagrams, conceptual illustrations, and documentation images.
* **`reports/`**: milestone reports, internal summaries, and progress writeups.

---

### `src/`

Contains the main source code of the project.

* **`core/`**: core abstractions, shared logic, and foundational components.
* **`data/`**: data loading, preprocessing, transformation, and preparation logic.
* **`models/`**: model architectures, layers, and model-related implementation.
* **`funcs/`**: mathematical functions, losses, metrics, and feature-related operations.
* **`services/`**: orchestration logic for training, inference, pipelines, or application services.
* **`api/`**: API routes, interface contracts, or service endpoints.
* **`backend/`**: backend application logic and system-side integration code.
* **`helpers/`**: utility functions, reusable helpers, and supporting code.

---

### `data/`

Contains project data organized by role in the pipeline.

* **`downloads/`**: original downloaded datasets or externally obtained raw resources.
* **`imports/`**: imported and standardized data ready to enter the project workflow.
* **`schemas/`**: structural definitions for datasets, labels, and expected formats.
* **`metadata/`**: provenance, annotations, descriptions, and dataset documentation.
* **`exports/`**: processed datasets, derived subsets, and data prepared for reuse or sharing.

---

### `models/`

Contains model artifacts and model-specific structured outputs.

* **`schemas/`**: model configurations, architecture definitions, and structural specifications.
* **`exports/`**: serialized models, deployment-ready artifacts, or shared model versions.
* **`benchmarks/`**: selected model variants prepared for comparison and evaluation.
* **`archive/`**: old checkpoints, deprecated model versions, and retired artifacts.

---

### `tests/`

Contains validation and quality assurance code.

* **`specs/`**: unit-level and behavior-level tests for isolated components.
* **`integration-tests/`**: tests that verify interactions between multiple components or subsystems.
* **`e2e/`**: end-to-end tests that validate complete workflows.
* **`benchmarks/`**: performance, regression, and repeatability checks.

---

### `scripts/`

Contains executable scripts used to operate the project.

* **`jobs/`**: scripts for training, evaluation, inference, and scheduled experiment runs.
* **`tools/`**: maintenance utilities, inspection scripts, and one-off operational helpers.
* **`gens/`**: scripts that generate files, templates, artifacts, or scaffolding.
* **`release/`**: packaging, export, and release preparation scripts.

---

### `configs/`

Contains configuration and operational setup files.

* **`envs/`**: environment definitions for local, server, or lab execution.
* **`docker/`**: container-related configuration and Docker setup.
* **`ci/`**: continuous integration and automation pipeline configuration.
* **`settings/`**: runtime settings, experiment parameters, and application configuration.

---

### `reports/`

Contains generated outputs and result artifacts.

* **`figs/`**: plots, graphs, and visual outputs generated from experiments or analyses.
* **`benchmarks/`**: benchmark results, comparison outputs, and evaluation summaries.
* **`exports/`**: publication-ready or presentation-ready result bundles.
* **`archive/`**: older reports, superseded outputs, and preserved historical results.

---

## Design Principles

This structure is built around a few simple ideas:

* **Documentation is separate from implementation**
  Writing and explanation live in `docs/`, while executable logic lives in `src/`.

* **Data, models, and results are separated**
  This keeps the workflow traceable and prevents project artifacts from getting mixed together.

* **Research and engineering can coexist cleanly**
  The structure supports both scientific writing and production-style software development.

* **Results are reproducible and easy to locate**
  Configurations, scripts, tests, and reports each have a clear place.

---

## Recommended Usage

As a general rule:

* put **code** in `src/`
* put **datasets and data definitions** in `data/`
* put **trained/exported model artifacts** in `models/`
* put **tests and validation logic** in `tests/`
* put **execution scripts** in `scripts/`
* put **environment and runtime configuration** in `configs/`
* put **generated results** in `reports/`
* put **notes, papers, and thesis material** in `docs/`

---

## Goal

The goal of this structure is to keep the project:

* **professional**
* **minimal**
* **research-ready**
* **AI-lab-ready**
* **easy to navigate**
* **visually clean in VS Code**

In other words: less folder chaos, more signal.
