# PSDCodec

**PSDCodec** is a neural codec for **power spectral density (PSD) compression**.

This repository is intended to support **both research and production-style engineering**:
- development of the codec itself,
- experimentation and benchmarking,
- paper/thesis writing,
- and deployment of the encoder on a resource-constrained embedded target such as a **Raspberry Pi 5**.

This `README.md` is the **project’s north star**: it explains what the project is, what the implementation boundaries are, how training and deployment should be organized, and how the repository should evolve.

For the detailed folder-level repository map, see **`STRUCTURE.md`**. This README does not replace it; it gives the **operational and architectural meaning** of that structure.

---

## 1. Project purpose

The goal of PSDCodec is to compress PSD frames into a compact discrete representation that can be transmitted under a bitrate budget and reconstructed later with useful fidelity for downstream sensing tasks.

The codec is built around four core ideas:

1. **Deterministic preprocessing** transforms the raw PSD frame into a representation that is easier to compress.
2. A **learned encoder** maps the preprocessed frame into latent vectors.
3. The latent vectors are **vector-quantized** into discrete codebook indices.
4. Those indices are **entropy coded** for actual transmission efficiency.

The system is therefore **not just an autoencoder**. It is a full **rate–distortion codec** with a clear distinction between:
- **training-time surrogate rate**, and
- **deployment-time operational payload**.

---

## 2. The engineering thesis of this repository

This repository should follow one central implementation principle:

> **Train the full codec in PyTorch, but deploy only the inference-time encoder with ONNX Runtime on the embedded device.**

That principle exists for a reason.

During training, the full codec needs machinery that belongs naturally in PyTorch:
- encoder,
- decoder,
- codebook,
- entropy model,
- rate–distortion objective,
- and straight-through estimation for vector quantization.

During deployment on a constrained device, the edge node should do only what is operationally necessary:
- deterministic preprocessing,
- encoder forward pass,
- nearest-codeword lookup,
- arithmetic coding,
- and payload transmission.

The **decoder should remain on the remote/server side** unless a future milestone explicitly requires local reconstruction.

This keeps the embedded system lean, testable, and realistic.

---

## 3. System boundary: what runs where

### Training / research environment

Use **PyTorch** as the source-of-truth framework for model development and experimentation.

The training graph includes:
- preprocessing-aware input handling,
- encoder `E_theta`,
- vector quantization via nearest-codeword assignment,
- decoder `G_phi`,
- entropy model for rate estimation,
- reconstruction and task-aware losses,
- VQ stabilization terms,
- optimizer and experiment logic.

### Embedded sensing node

Use **ONNX Runtime** for **encoder inference only**.

The sensing node should perform:
- PSD frame acquisition,
- deterministic preprocessing,
- encoder inference,
- nearest-codeword search,
- arithmetic coding of discrete indices,
- packing of side information,
- transmission to the remote side.

### Remote server / backend

The remote side should perform:
- arithmetic decoding,
- codebook lookup,
- decoder inference,
- inverse preprocessing,
- storage, visualization, or downstream task consumption.

### Non-negotiable boundary

The following should **stay outside** the exported ONNX graph:
- deterministic preprocessing,
- side-information handling,
- nearest-codeword lookup,
- arithmetic coding,
- deployment packetization logic.

The exported ONNX model should ideally be **only the encoder forward pass**.

That boundary is the single most important architectural rule in this project.

---

## 4. End-to-end workflow

```text
Raw PSD frame
   -> deterministic preprocessing
   -> encoder (PyTorch during research / ONNX Runtime during deployment)
   -> latent vectors
   -> nearest-codeword vector quantization
   -> discrete index stream
   -> arithmetic coding + side information packing
   -> transmitted payload
   -> remote arithmetic decoding + codebook lookup
   -> decoder
   -> inverse preprocessing
   -> reconstructed PSD
```

A practical mental model:
- **PyTorch owns learning**
- **ONNX Runtime owns embedded inference**
- **plain Python / NumPy / native code owns deterministic codec logic**

---

## 5. Repository north-star structure

The repository already has a strong structure. The role of each major directory in the PSDCodec plan should be as follows.

## `docs/`
Human-readable material.

Use this for:
- research notes,
- paper and thesis writing,
- derivations,
- diagrams,
- milestone writeups,
- experiment interpretation.

`docs/` is where the project explains itself.

## `src/`
Executable source code.

Recommended PSDCodec mapping:

- `src/core/`  
  Core abstractions for frames, packets, codec interfaces, and shared data structures.

- `src/data/`  
  PSD ingestion, dataset loading, preprocessing transforms, dataset splits, and batching.

- `src/models/`  
  Encoder, decoder, codebook modules, entropy model components, and export wrappers.

- `src/funcs/`  
  Distortion functions, rate terms, VQ losses, occupancy/task metrics, and mathematical utilities.

- `src/services/`  
  Training loops, evaluation pipelines, export services, and deployment orchestration.

- `src/api/`  
  API contracts if the remote server exposes codec or reconstruction services.

- `src/backend/`  
  Remote reconstruction pipeline, server-side decoding flow, persistence, task-serving logic.

- `src/helpers/`  
  Logging, configuration loading, reproducibility helpers, filesystem utilities, and misc support code.

## `data/`
Datasets and dataset lineage.

Use this to separate:
- raw downloads,
- imported/standardized PSD data,
- schemas,
- metadata,
- processed exports.

The data folder must preserve **traceability**. A future you should never have to wonder where a dataset came from or what transformation produced it.

## `models/`
Model artifacts.

Use this for:
- saved checkpoints,
- exported ONNX models,
- benchmark-ready variants,
- archived experiments.

A useful convention:
- training checkpoints live in `models/archive/` or a structured experiment subfolder,
- deployment-ready ONNX exports live in `models/exports/`,
- comparison-ready named variants live in `models/benchmarks/`.

## `tests/`
Quality gates.

Recommended split:
- `tests/specs/`: unit tests for losses, transforms, codebook lookup, entropy-model pieces,
- `tests/integration-tests/`: PyTorch training/evaluation flow, export flow, packet encode/decode consistency,
- `tests/e2e/`: full frame -> payload -> reconstruction workflow,
- `tests/benchmarks/`: latency, memory, throughput, and regression tracking.

## `scripts/`
Operational entry points.

Recommended usage:
- `scripts/jobs/`: training, evaluation, export, and benchmark runs,
- `scripts/tools/`: inspection and debugging utilities,
- `scripts/gens/`: artifact/config/template generation,
- `scripts/release/`: packaging models and deployment bundles.

## `configs/`
Configuration and reproducibility.

This should contain:
- environment specs,
- CI definitions,
- runtime settings,
- model/training/deployment config files,
- hardware-specific settings for the Pi and server.

## `reports/`
Generated outputs.

This is where the project proves what happened:
- figures,
- benchmark outputs,
- exported summaries,
- archived results.

---

## 6. Training plan

The training plan should be implemented in layers.

### Phase A — data and preprocessing foundation

Build the deterministic input pipeline first.

Deliverables:
- PSD frame loader,
- reproducible preprocessing transforms,
- side-information extraction and quantization rules,
- preprocessing-only reconstruction reference,
- train/val/test split definition,
- dataset metadata tracking.

This phase matters because codec performance becomes meaningless if the data path is unstable or leaky.

### Phase B — baseline learned codec in PyTorch

Implement the initial full codec in PyTorch:
- encoder,
- decoder,
- codebook,
- nearest-codeword assignment,
- straight-through gradient handling,
- reconstruction losses,
- VQ stabilization losses,
- entropy-model rate proxy.

Deliverables:
- a trainable baseline,
- a reproducible config file for the baseline,
- first rate–distortion curves,
- validation plots and error diagnostics.

### Phase C — task-aware optimization

Extend the loss from pure reconstruction to application-aware performance.

Possible deliverables:
- occupancy-related metrics,
- downstream task losses,
- ablation studies on distortion terms,
- comparison against preprocessing-only and simpler baselines.

### Phase D — operational rate alignment

Close the gap between training-time rate proxy and real deployment payload.

Deliverables:
- arithmetic coder implementation,
- consistency checks between entropy-model estimates and actual coded length,
- reporting based on **operational payload**, not just surrogate rate,
- packet format definition.

### Phase E — deployment export path

Once the encoder is stable, export it to ONNX and validate numerical parity.

Deliverables:
- deterministic encoder export script,
- parity tests between PyTorch and ONNX Runtime,
- deployment-ready encoder artifact in `models/exports/`,
- documented model I/O contract.

### Phase F — Raspberry Pi 5 integration

Deploy the edge-side path on the target embedded platform.

Deliverables:
- ONNX Runtime session on the Pi,
- preprocessing + encoder + VQ + arithmetic coding pipeline,
- memory and latency benchmark results,
- reproducible deployment script or service.

---

## 7. Deployment plan

The deployment plan should be simple, explicit, and boring in the best way.

### Embedded target

Primary target:
- **Raspberry Pi 5**

Edge responsibilities:
- receive or compute PSD frame,
- preprocess deterministically,
- run encoder using ONNX Runtime,
- quantize to codebook indices,
- entropy code payload,
- transmit compact packet.

### Runtime choice

Use **ONNX Runtime** for the embedded inference layer because it offers:
- a clean inference-only runtime,
- good documentation,
- practical ARM support,
- a deployment path that is much lighter than dragging a full training framework into the Pi.

### Export policy

Only export the **encoder**.

Do **not** export the following into the ONNX graph unless there is a compelling technical reason and a benchmark proving it helps:
- arithmetic coding,
- codebook search,
- preprocessing transforms with complicated side effects,
- server-side reconstruction logic.

### Server-side responsibilities

The remote/backend side should remain flexible and easier to evolve than the edge node.

Recommended server responsibilities:
- decode packet,
- reconstruct PSD,
- run downstream analysis,
- expose storage/API/reporting hooks,
- support benchmark and visualization workflows.

---

## 8. Validation strategy

This project should be validated at four levels.

### 1. Mathematical correctness

Verify that:
- preprocessing and inverse preprocessing are internally consistent,
- rate computations are dimensionally and operationally correct,
- codebook lookup and index handling are deterministic.

### 2. Learning behavior

Verify that:
- losses decrease sensibly,
- the codebook is being used rather than collapsing,
- the entropy model behaves coherently,
- reconstructed PSDs preserve the features that matter.

### 3. Export safety

Verify that:
- encoder export succeeds reproducibly,
- ONNX Runtime output matches PyTorch within tolerance,
- input/output tensor contracts are versioned and documented.

### 4. Embedded deployment behavior

Verify that:
- inference latency is acceptable on the Raspberry Pi 5,
- memory usage is stable,
- payload generation works end-to-end,
- actual transmitted bitrate matches expectations.

---

## 9. Benchmark philosophy

This project should report **operationally meaningful** results.

That means benchmarks must not stop at “the loss went down.”

The important benchmark families are:
- reconstruction quality,
- task-aware quality,
- bitrate / payload size,
- latency,
- memory footprint,
- export parity,
- embedded throughput,
- robustness across datasets or sensing conditions.

A result is not complete unless it answers both:
1. **How good is the reconstruction/task performance?**
2. **What did it cost in actual transmitted bits and runtime resources?**

---

## 10. Milestones

A sensible milestone sequence for the repository is:

### Milestone 1 — preprocessing pipeline
Stable PSD ingestion and deterministic transforms.

### Milestone 2 — PyTorch baseline codec
End-to-end training with reconstruction and VQ loss.

### Milestone 3 — rate-aware codec
Entropy model and operational payload accounting.

### Milestone 4 — evaluation and ablations
Rate–distortion curves, baseline comparisons, and task-aware experiments.

### Milestone 5 — ONNX export path
Encoder-only export with parity tests.

### Milestone 6 — Raspberry Pi deployment
On-device inference plus packet generation.

### Milestone 7 — server reconstruction stack
Operational remote decoding and reconstruction service.

### Milestone 8 — publication-ready results
Clean figures, reproducible tables, documented conclusions.

---

## 11. Development rules

To keep this repository sane, follow these rules.

### Rule 1
**Do not mix research notes with executable code.**  
Ideas belong in `docs/`; implementation belongs in `src/`.

### Rule 2
**Do not treat training-time surrogate rate as deployment truth.**  
Always distinguish estimated rate from arithmetic-coded operational payload.

### Rule 3
**Do not over-export.**  
The ONNX graph should stay minimal. Export the encoder unless a benchmark-backed reason says otherwise.

### Rule 4
**Do not bury reproducibility in tribal memory.**  
Configs, scripts, reports, and artifacts should be versioned and discoverable.

### Rule 5
**Do not let embedded constraints enter too late.**  
Track latency, memory, and payload structure before the project is “finished.”

### Rule 6
**Every important result should be reproducible from code, config, and data lineage.**

---

## 12. Suggested implementation conventions

These are recommended conventions for consistency.

### Naming
- use `encoder`, `decoder`, `codebook`, `entropy_model`, `preprocess`, `packet`, `reconstruct` as plain internal names,
- keep experiment names short but descriptive,
- separate `train`, `eval`, `export`, and `deploy` concerns clearly.

### Artifact organization
- checkpoint names should include config identity and date/version,
- ONNX exports should include model variant and opset/version,
- benchmark reports should record hardware, config, and dataset split.

### Configuration
Prefer explicit config files over hardcoded constants.

At minimum, configs should define:
- data source and split,
- preprocessing parameters,
- encoder/decoder architecture,
- codebook size and latent dimensions,
- loss weights,
- optimizer settings,
- export settings,
- deployment settings.

---

## 13. What success looks like

PSDCodec is successful when the repository can do all of the following coherently:

1. train a rate-aware PSD codec reproducibly,
2. evaluate it with meaningful rate–distortion and task-aware metrics,
3. export the encoder safely to ONNX,
4. run the edge pipeline on a Raspberry Pi 5,
5. reconstruct remotely with a clean server-side flow,
6. generate publication-ready figures and reports,
7. remain understandable to a future contributor without archaeology.

That last one matters more than people admit.

---

## 14. Reading order for contributors

If you are new to the repository, read in this order:

1. `README.md` — project intent and engineering plan
2. `STRUCTURE.md` — repository layout and folder semantics
3. `docs/notes/` and `docs/articles/` — research context and theory
4. `configs/` — experiment and runtime configuration
5. `src/` — implementation
6. `tests/` — verification strategy
7. `reports/` — evidence and results

---

## 15. Final project stance

This repository is meant to be both:
- a **serious research project**, and
- a **deployable codec engineering effort**.

That means the code should be mathematically grounded, experimentally reproducible, and operationally honest.

The north-star decision is simple:

> **Use PyTorch to learn the codec. Use ONNX Runtime to deploy the encoder. Keep the rest of the codec logic explicit and testable.**

That is the plan.
