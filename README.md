# PSDCodec

PSDCodec is a codec-first research engineering repository for compressing sampled
power spectral density (PSD) frames under explicit bitrate constraints.

The project treats PSD transport as an end-to-end codec problem rather than only
as a reconstruction network. Deterministic preprocessing, side information,
vector quantization, entropy coding, deployment payload, and reconstruction
quality are all modeled explicitly.

## Scope

At each time index `t`, a sensing node observes a PSD frame over a monitored
frequency band, converts it into a compact payload, and transmits that payload to
a remote server. The current repository implements the three-part framework
defined in [`docs/articles/main.tex`](/home/marti/Code/PSDCodec/docs/articles/main.tex):

1. Deterministic preprocessing with explicit side information.
2. A learned discrete latent codec with vector quantization.
3. Operational rate accounting plus PSD-domain and optional task-aware distortion.

The deployment split is documented in
[`docs/notes/DEPLOYMENT_BOUNDARY.md`](/home/marti/Code/PSDCodec/docs/notes/DEPLOYMENT_BOUNDARY.md):
PyTorch owns full research-time training and validation, while ONNX Runtime is
used only for the encoder boundary on the sensing node.

## What Is Implemented

The current demo is materially beyond a placeholder prototype. The repository
contains:

- Deterministic preprocessing and inverse preprocessing with quantized blockwise
  side information in [`src/codec/preprocessing.py`](/home/marti/Code/PSDCodec/src/codec/preprocessing.py).
- Explicit scalar/vector quantization, factorized entropy modeling, arithmetic
  coding, and packet serialization in
  [`src/codec/`](/home/marti/Code/PSDCodec/src/codec).
- A PyTorch convolutional learned codec, training-time VQ loss, and ONNX encoder
  export in [`src/models/torch_backend.py`](/home/marti/Code/PSDCodec/src/models/torch_backend.py).
- Raw campaign ingestion, harmonization, prepared dataset caching, and dataset
  contracts in [`src/data/`](/home/marti/Code/PSDCodec/src/data).
- Deployment-oriented evaluation, export loading, and notebook support in
  [`src/interfaces/deployment.py`](/home/marti/Code/PSDCodec/src/interfaces/deployment.py)
  , [`src/interfaces/evaluation.py`](/home/marti/Code/PSDCodec/src/interfaces/evaluation.py),
  and [`src/interfaces/demo_animation.py`](/home/marti/Code/PSDCodec/src/interfaces/demo_animation.py).
- A canonical manuscript-backed demo experiment in
  [`configs/experiments/demo.yaml`](/home/marti/Code/PSDCodec/configs/experiments/demo.yaml),
  [`scripts/jobs/train_demo.py`](/home/marti/Code/PSDCodec/scripts/jobs/train_demo.py),
  and [`notebooks/demo_deploy.ipynb`](/home/marti/Code/PSDCodec/notebooks/demo_deploy.ipynb).
- Unit and integration coverage across preprocessing, entropy coding, training,
  export recovery, deployment evaluation, and notebook-facing helpers under
  [`tests/`](/home/marti/Code/PSDCodec/tests).

## Recommended Reading Order

To understand the repo in the right order:

1. Read [`docs/articles/main.tex`](/home/marti/Code/PSDCodec/docs/articles/main.tex)
   or [`docs/articles/main.pdf`](/home/marti/Code/PSDCodec/docs/articles/main.pdf)
   for the mathematical framework.
2. Read
   [`docs/notes/DEPLOYMENT_BOUNDARY.md`](/home/marti/Code/PSDCodec/docs/notes/DEPLOYMENT_BOUNDARY.md)
   for the encoder-only ONNX split.
3. Read [`STRUCTURE.md`](/home/marti/Code/PSDCodec/STRUCTURE.md) for the actual
   repository contract and current file-placement assessment.
4. Inspect the canonical demo configuration at
   [`configs/experiments/demo.yaml`](/home/marti/Code/PSDCodec/configs/experiments/demo.yaml).

## Canonical Entry Points

- Train the generic YAML-driven pipeline:
  [`scripts/jobs/train_codec.py`](/home/marti/Code/PSDCodec/scripts/jobs/train_codec.py)
- Train the canonical demo:
  [`scripts/jobs/train_demo.py`](/home/marti/Code/PSDCodec/scripts/jobs/train_demo.py)
- Prepare a processed dataset from raw campaigns:
  [`scripts/jobs/prepare_campaign_dataset.py`](/home/marti/Code/PSDCodec/scripts/jobs/prepare_campaign_dataset.py)
- Recover an export bundle from a saved checkpoint:
  [`scripts/jobs/recover_codec_export.py`](/home/marti/Code/PSDCodec/scripts/jobs/recover_codec_export.py)
- Generate the formal rate-distortion-complexity report for the demo:
  [`scripts/tools/demo_eval.py`](/home/marti/Code/PSDCodec/scripts/tools/demo_eval.py)
- Inspect the deployment workflow:
  [`notebooks/demo_deploy.ipynb`](/home/marti/Code/PSDCodec/notebooks/demo_deploy.ipynb)

## Validation Commands

The repository currently exposes these useful checks:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src tests scripts
```

Current baseline:

- `pytest` passes.
- `ruff check .` passes.
- `ruff format --check .` passes.
- `mypy src tests scripts` passes.

The repository also now exposes a generated deployment-oriented benchmark at
[`reports/benchmarks/demo_eval.md`](/home/marti/Code/PSDCodec/reports/benchmarks/demo_eval.md)
with its machine-readable companion
[`reports/benchmarks/demo_eval.json`](/home/marti/Code/PSDCodec/reports/benchmarks/demo_eval.json).

## Current Repository State

The implementation is strongest where the project is most opinionated:

- Theory-to-code traceability is unusually good for a demo.
- The codec boundary is explicit rather than hidden inside one opaque model.
- Deployment analysis is treated as a first-class concern.
- The test suite covers the main operational paths, including checkpoint/export
  recovery and ONNX deployment round-trips.

The main weaknesses are repository hygiene rather than missing core ideas:

- The root `README.md` and `STRUCTURE.md` had drifted out of place and are now
  restored from the legacy drafts.
- Generated artifacts may exist locally under
  [`data/processed/`](/home/marti/Code/PSDCodec/data/processed),
  [`models/checkpoints/`](/home/marti/Code/PSDCodec/models/checkpoints), and
  [`models/exports/`](/home/marti/Code/PSDCodec/models/exports), but they are now
  treated as reproducible outputs rather than versioned source.
- Packaging metadata under `*.egg-info/` is treated as generated output and is no
  longer part of the intended source tree.
- Several directories are intentionally reserved but still empty, which is fine as
  long as they remain placeholders rather than silent dumping grounds.

## Artifact Policy Note

The repository now takes the stricter position:

- processed dataset caches are reproducible outputs,
- demo checkpoints and export bundles are reproducible outputs,
- ONNX/runtime bundles are generated deployment artifacts,
- and packaging metadata such as `*.egg-info/` is generated build output.

That keeps the repository centered on theory, source, tests, and configuration.
The canonical way to recreate demo outputs is through
[`scripts/jobs/train_demo.py`](/home/marti/Code/PSDCodec/scripts/jobs/train_demo.py)
and the export-recovery utilities, not through committed binaries.

## License

PSDCodec is released under the MIT License. See
[`LICENSE`](/home/marti/Code/PSDCodec/LICENSE).
