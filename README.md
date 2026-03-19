# PSDCodec

**PSDCodec** is a codec-first research engineering project for **compressing sampled power spectral density (PSD) frames under explicit bitrate constraints**.

A sensing node estimates a PSD over a monitored frequency band, transmits a compact representation, and a remote server reconstructs the spectrum for storage, analysis, or downstream spectrum-sensing tasks. The project is framed as a real codec problem: bitrate, side information, reconstruction quality, and deployment payload are treated as first-class concerns.

---

## Overview

PSDCodec is organized around three conceptual layers:

1. **Deterministic preprocessing** of PSD frames before learning.
2. **Learned discrete latent coding** for compact representation.
3. **Rate-distortion evaluation** based on operational payload rather than reconstruction error alone.

The repository supports both the **mathematical framework** of the codec and the **software infrastructure** needed to implement, test, benchmark, and eventually deploy it.

---

## Why this project exists

Many learned compression systems are presented primarily as reconstruction networks. PSDCodec takes a stricter stance:

- it treats the system as a **codec**, not only as an autoencoder,
- it keeps **deterministic simplification** separate from learned compression,
- it makes **bitrate accounting explicit**, including side information,
- and it distinguishes among **signal fidelity**, **task fidelity**, and **deployment payload**.

That position gives the repository a clear technical identity: the goal is not merely to reconstruct PSDs, but to do so under an honest compression model that can be analyzed, ablated, and benchmarked rigorously.

---

## How to read this repository

For a new reader, the cleanest entry path is:

1. Read **`README.md`** for project mission, scope, and repository-level orientation.
2. Read **`STRUCTURE.md`** for folder semantics and repository boundaries.
3. Read **`docs/articles/main.pdf`** for the mathematical framework and codec formulation.
4. Read **`docs/notes/DEPLOYMENT_BOUNDARY.md`** for the encoder-only ONNX deployment split.

This README is the front door. `STRUCTURE.md` defines repository organization. `docs/articles/main.pdf` is the main theoretical reference.

---

## Repository layout

The tree below is a top-level summary only; **`STRUCTURE.md`** remains the normative repository contract.

```text
psdcodec/
├── docs/       # notes, articles, figures, reports
├── src/        # implementation code
├── data/       # datasets, schemas, metadata, exports
├── models/     # model artifacts, exports, benchmarks, archive
├── tests/      # specs, integration, e2e, benchmarks
├── scripts/    # jobs, tools, generators, release utilities
├── configs/    # environments, docker, CI, settings
└── reports/    # generated figures, benchmarks, exports, archive
```

At a high level:

- **`docs/`** contains human-readable project material.
- **`src/`** contains executable implementation.
- **`data/`**, **`models/`**, and **`reports/`** keep inputs, artifacts, and outputs separate.
- **`tests/`**, **`scripts/`**, and **`configs/`** support validation and operation.

---

## Available now

The repository now implements the full codec workflow described in the manuscript:

- deterministic preprocessing and inverse preprocessing with explicit side information,
- vector quantization and arithmetic coding for the latent index stream,
- a PyTorch training pipeline with checkpointing and encoder export,
- raw campaign ingestion from **`data/raw/campaigns/`**,
- deployment helpers that keep the encoder-only ONNX boundary explicit,
- and notebooks for deployment inspection and the manuscript's illustrative task demo.

The repository is intentionally kept **artifact-clean**: no trained checkpoints or
exported models are committed. The canonical first run is the manuscript-backed
**`demo`** experiment, which trains the illustrative sensing task described in
**`docs/articles/main.tex`** and exports its deployment assets under
**`models/exports/demo/`**.

The main operational entrypoints are:

- **`scripts/jobs/prepare_campaign_dataset.py`** for harmonizing raw campaign PSD acquisitions,
- **`scripts/jobs/train_codec.py`** for YAML-driven training,
- **`scripts/jobs/train_demo.py`** for the canonical manuscript-backed demo model,
- **`notebooks/demo_deploy.ipynb`** for the complete deployment and illustrative-task demo.

## Theory-to-Code Map

The manuscript is the normative mathematical reference. The implementation follows that structure directly:

- **Deterministic preprocessing and side information**
  - **`src/codec/preprocessing.py`**
  - **`src/codec/quantization.py`**
  - **`src/codec/torch_preprocessing.py`**
- **Learned discrete latent codec**
  - **`src/models/torch_backend.py`**
  - **`src/models/reference.py`**
- **Operational rate model and payload accounting**
  - **`src/codec/entropy.py`**
  - **`src/codec/arithmetic.py`**
  - **`src/codec/packetization.py`**
  - **`src/pipelines/runtime.py`**
- **PSD distortion and illustrative sensing task**
  - **`src/objectives/distortion.py`**
  - **`src/objectives/training.py`**
- **Training, export, and deployment boundary**
  - **`src/pipelines/training.py`**
  - **`src/interfaces/export.py`**
  - **`src/interfaces/deployment.py`**
  - **`docs/notes/DEPLOYMENT_BOUNDARY.md`**

---

## Near-term implementation priorities

Remaining priorities are no longer the basic codec itself, but the next layer of rigor:

- stronger training-time and deployment-time result reporting,
- optional richer entropy models beyond the current factorized baseline,
- and continued tightening of theory-to-implementation traceability as the codec evolves.

---

## Working principles

PSDCodec is organized around a few early rules:

- keep **domain logic** separate from orchestration and utilities,
- keep **research writing** separate from executable code,
- make **rate and evaluation assumptions explicit**, and
- prefer changes that improve **reproducibility, interpretability, and testability**.

A good contribution is not only one that improves a metric, but one that makes the codec easier to reason about and easier to evaluate honestly.

---

## Core references

The current core references are:

- **`README.md`** — repository-level orientation
- **`STRUCTURE.md`** — directory semantics and organization contract
- **`docs/articles/main.tex` / `docs/articles/main.pdf`** — mathematical framework and manuscript source
- **`docs/notes/DEPLOYMENT_BOUNDARY.md`** — deployment split and ONNX scope

As the project grows, additional design documents may be added for topics such as deployment boundaries, runtime interfaces, and evaluation policy.

---

## License

This project is released under the **MIT License**. See **`LICENSE`** for details.

---

## In one line

**PSDCodec is a codec-first repository for PSD compression under explicit bitrate constraints, designed to support rigorous theory, disciplined implementation, and reproducible evaluation.**
