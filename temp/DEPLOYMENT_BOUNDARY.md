# Deployment Boundary

This document defines the deployment boundary for **PSDCodec**.

PSDCodec separates **research-time learning** from **deployment-time execution**. The full codec is developed, trained, and validated in **PyTorch**. During deployment, **ONNX Runtime** executes only the exported inference-time encoder on constrained hardware. Deterministic preprocessing, side-information handling, nearest-codeword assignment, packetization, entropy coding, and reconstruction remain outside the exported graph by design.

---

## Boundary Statement

- **PyTorch** is the environment for developing, training, and validating the full codec.
- **ONNX Runtime** is the deployment surface for encoder inference only.
- **Explicit codec software outside ONNX** owns preprocessing, quantization-related assignment logic, side-information handling, packetization, entropy coding, and reconstruction.

Operationally:

- **PyTorch** = full learning system
- **ONNX Runtime** = embedded encoder inference only
- **Native codec software outside ONNX** = deterministic and protocol-level compression logic

---

## Why this boundary exists

PSDCodec is treated as a **codec system**, not as a single opaque neural graph.

This boundary exists to preserve three properties at once:

- **research flexibility** for the full learned codec during training and validation,
- **deployment efficiency** on the sensing node,
- **software transparency and testability** for deterministic codec logic.

In particular:

- deterministic preprocessing remains explicit and inspectable,
- vector-quantization behavior remains under direct software control,
- entropy coding and packetization remain codec responsibilities rather than graph internals,
- the embedded deployment payload stays narrow,
- and learned inference can evolve without absorbing the entire transport and reconstruction stack into the exported model.

---

## PyTorch Scope

PyTorch is the research and validation environment for the complete codec.

Within PyTorch, the project trains and evaluates the full pipeline:

1. deterministic preprocessing of PSD frames,
2. learned encoding into latent representations,
3. vector quantization with the codebook,
4. learned decoding of the normalized representation,
5. PSD-space distortion evaluation,
6. explicit rate accounting,
7. optional task-aware loss terms,
8. and vector-quantization stabilization during training.

PyTorch therefore owns:

- model development,
- training,
- ablation,
- validation,
- export preparation,
- and controlled comparison of codec variants.

PyTorch is not the embedded deployment surface.

---

## ONNX Runtime Scope

ONNX Runtime is the deployment surface for **encoder inference only**.

The exported artifact is the inference-time encoder that transforms an already preprocessed PSD representation into the latent representation **before nearest-codeword assignment**. At that handoff point, ownership passes from the exported neural model to explicit codec software.

This deployment boundary is intentionally narrow:

- ONNX Runtime runs on the embedded sensing node.
- The exported graph performs encoder inference only.
- The exported graph does not contain the full codec.

The primary target envisioned by this boundary is an embedded device such as a **Raspberry Pi 5**, where efficient inference matters but codec logic must remain explicit.

---

## What remains outside ONNX

The following logic remains outside the exported ONNX graph by design.

### Edge-side logic

On the sensing node, explicit software is responsible for:

- deterministic preprocessing,
- side-information handling,
- nearest-codeword assignment,
- packetization,
- and arithmetic coding.

These steps remain explicit so they can be inspected, tested, and modified without changing the exported neural artifact.

### Server-side logic

On the reconstruction side, explicit software is responsible for:

- decoding the transmitted representation,
- recovering the quantized latent representation and side information,
- learned or codec-controlled reconstruction of the normalized representation,
- inverse standardization,
- inverse mapping,
- and upsampling back to PSD form.

These steps belong to the end-to-end codec pipeline, but they are not part of the embedded ONNX deployment surface.

---

## End-to-end operational split

The intended operational workflow is:

### On the sensing node

1. Acquire or receive the PSD frame.
2. Apply deterministic preprocessing.
3. Run the exported encoder with ONNX Runtime.
4. Perform nearest-codeword assignment.
5. Attach required side information.
6. Packetize and entropy-code the payload.
7. Transmit the compressed representation.

### On the remote server

1. Receive and decode the payload.
2. Recover the quantized latent representation and side information.
3. Reconstruct the normalized representation.
4. Apply inverse standardization and inverse mapping.
5. Upsample as required.
6. Recover the final PSD estimate.

This matches the project workflow described in the theoretical framework: the sensing node preprocesses and encodes, while the remote server reconstructs the PSD.

---

## Architectural rationale

This boundary gives the project several engineering benefits.

### 1. Smaller deployment surface

Only the encoder is exported and executed on the embedded target. This keeps the runtime artifact narrow and efficient.

### 2. Transparent codec logic

Preprocessing, quantization-related assignment, side-information handling, and arithmetic coding remain readable software rather than opaque graph internals.

### 3. Better testability

The learned component and the deterministic codec components can be validated separately and together. Failures are easier to localize.

### 4. Clearer software ownership

Neural inference belongs to the exported model. Codec protocol logic belongs to explicit source code. This keeps responsibilities auditable.

### 5. Easier iteration

Changes to packet format, preprocessing details, assignment policy, or entropy coding do not require redefining the deployment graph.

---

## Current exclusions

The current deployment boundary explicitly excludes:

- deploying the full end-to-end codec as a single ONNX graph,
- embedding deterministic preprocessing inside the exported model artifact,
- moving entropy coding into the neural deployment graph,
- and treating server-side reconstruction as part of the embedded inference surface.

These exclusions are part of the present design boundary, not incidental implementation omissions.

---

## Repository implications

This boundary has direct consequences for repository organization.

- **Training and validation code** belong with the full learning pipeline.
- **Export logic** produces encoder-only deployment artifacts.
- **Deterministic codec logic** remains implemented as explicit source modules.
- **Runtime interfaces** reflect the split between embedded encoding and server-side reconstruction.
- **Tests** validate both the isolated encoder export and the explicit codec stages around it.

As the repository grows, this boundary should remain visible in code organization, testing strategy, export procedures, and deployment documentation.

---

## In one line

**PSDCodec is a codec-first system in which the full codec is trained in PyTorch, only the inference-time encoder is deployed with ONNX Runtime, and deterministic codec logic remains outside the exported graph under direct software control.**
