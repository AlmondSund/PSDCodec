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
3. Read **`main.pdf`** for the mathematical framework and codec formulation.

This README is the front door. `STRUCTURE.md` defines repository organization. `main.pdf` is the main theoretical reference.

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

The strongest part of the repository today is its **theoretical and architectural foundation**.

Current materials already define:

- the codec problem setting for PSD frame compression,
- deterministic preprocessing and inverse preprocessing,
- discrete latent coding with vector quantization,
- explicit rate accounting,
- task-aware evaluation as an optional extension,
- and a repository structure designed for disciplined research software.

What is **not yet formally declared** by the current repository materials is a canonical installation path, package manager, CLI surface, or stable runtime entrypoint. This README therefore does not invent commands or workflows that the repository has not yet specified.

Today, a new contributor can productively read **`main.pdf`**, inspect **`STRUCTURE.md`**, and use those documents to guide implementation work while treating runtime entrypoints as still under definition.

---

## Near-term implementation priorities

As implementation matures, near-term priorities include:

- concrete preprocessing and reconstruction modules,
- trainable encoder/decoder/codebook components,
- entropy-model and arithmetic-coding integration,
- reproducible experiment and benchmarking scripts,
- clearer runtime entrypoints,
- and export-ready model artifacts.

This is the intended implementation direction, not a claim that all of those components already exist.

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
- **`main.tex` / `main.pdf`** — mathematical framework and manuscript source

As the project grows, additional design documents may be added for topics such as deployment boundaries, runtime interfaces, and evaluation policy.

---

## License

This project is released under the **MIT License**. See **`LICENSE`** for details.

---

## In one line

**PSDCodec is a codec-first repository for PSD compression under explicit bitrate constraints, designed to support rigorous theory, disciplined implementation, and reproducible evaluation.**
