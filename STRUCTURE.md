# Project Structure

This document defines the current repository contract for PSDCodec.

The goal is to keep theory, executable code, data lifecycle, trained artifacts,
and generated outputs structurally separate enough that the repository remains
auditable before new features are added. Unlike the earlier draft, this document
describes the repository as it actually exists today, including the places where
artifact hygiene still needs tightening.

## Root Layout

```text
psdcodec/
├── README.md
├── STRUCTURE.md
├── pyproject.toml
├── docs/
├── configs/
├── src/
├── tests/
├── scripts/
├── data/
├── models/
├── reports/
├── notebooks/
```

## Repository Roles

### `docs/`

Human-authored technical documentation.

- [`docs/articles/`](/home/marti/Code/PSDCodec/docs/articles) contains the
  manuscript source and rendered PDF.
- [`docs/notes/`](/home/marti/Code/PSDCodec/docs/notes) contains focused design
  notes such as the deployment boundary.
- [`docs/figs/`](/home/marti/Code/PSDCodec/docs/figs) and
  [`docs/progress/`](/home/marti/Code/PSDCodec/docs/progress) are currently empty
  placeholders and are structurally acceptable.

Rule:
`docs/` is for authored explanation, not generated experiment output.

### `configs/`

Declarative configuration.

- [`configs/experiments/demo.yaml`](/home/marti/Code/PSDCodec/configs/experiments/demo.yaml)
  is the canonical experiment configuration.
- [`configs/ci/`](/home/marti/Code/PSDCodec/configs/ci),
  [`configs/docker/`](/home/marti/Code/PSDCodec/configs/docker),
  [`configs/envs/`](/home/marti/Code/PSDCodec/configs/envs), and
  [`configs/runtime/`](/home/marti/Code/PSDCodec/configs/runtime) exist as empty
  placeholders.

Rule:
new runtime or experiment settings should be declared here instead of being baked
 into scripts or notebooks.

### `src/`

Executable source code.

- [`src/codec/`](/home/marti/Code/PSDCodec/src/codec):
  deterministic preprocessing, side information, quantization, entropy coding,
  arithmetic coding, and packetization.
- [`src/data/`](/home/marti/Code/PSDCodec/src/data):
  raw campaign ingestion, harmonization, prepared dataset construction, and batch
  collation.
- [`src/models/`](/home/marti/Code/PSDCodec/src/models):
  model contracts, NumPy reference model, and PyTorch training/export backend.
- [`src/objectives/`](/home/marti/Code/PSDCodec/src/objectives):
  PSD distortion, illustrative task metrics, and differentiable training losses.
- [`src/pipelines/`](/home/marti/Code/PSDCodec/src/pipelines):
  runtime orchestration, training, checkpointing, export, and selection policy.
- [`src/interfaces/`](/home/marti/Code/PSDCodec/src/interfaces):
  service boundary, deployment loading, notebook-facing reporting, and ONNX/PyTorch
  deployment bridging.
- [`src/utils/`](/home/marti/Code/PSDCodec/src/utils):
  narrow shared NumPy validation helpers.
- [`src/infra/`](/home/marti/Code/PSDCodec/src/infra) currently exists but is empty.

Rule:
the code boundaries are mostly well allocated. The main exception is
[`src/psdcodec.egg-info/`](/home/marti/Code/PSDCodec/src/psdcodec.egg-info),
which is generated packaging metadata and should usually not be treated as source.

### `tests/`

Automated validation.

- [`tests/specs/`](/home/marti/Code/PSDCodec/tests/specs) contains unit and
  component-level behavior tests.
- [`tests/integration/`](/home/marti/Code/PSDCodec/tests/integration) contains
  training, export, deployment, and end-to-end integration checks.
- [`tests/benchmarks/`](/home/marti/Code/PSDCodec/tests/benchmarks) and
  [`tests/e2e/`](/home/marti/Code/PSDCodec/tests/e2e) are currently empty.

Rule:
tests belong here, even when they validate notebook-facing or deployment-facing
helpers.

### `scripts/`

Operational entrypoints.

- [`scripts/jobs/`](/home/marti/Code/PSDCodec/scripts/jobs) contains the actual
  runnable training, export-recovery, and dataset-preparation CLIs.
- [`scripts/generate/`](/home/marti/Code/PSDCodec/scripts/generate),
  [`scripts/release/`](/home/marti/Code/PSDCodec/scripts/release), and
  [`scripts/tools/`](/home/marti/Code/PSDCodec/scripts/tools) are currently empty
  placeholders.

Rule:
scripts may orchestrate `src/` modules, but reusable logic should stay in `src/`.

### `data/`

Dataset assets and dataset lifecycle artifacts.

- [`data/raw/campaigns/`](/home/marti/Code/PSDCodec/data/raw/campaigns) is the raw
  campaign acquisition root.
- [`data/metadata/campaigns/INVENTORY.md`](/home/marti/Code/PSDCodec/data/metadata/campaigns/INVENTORY.md)
  documents the imported campaign corpus.
- `data/processed/` is reserved for generated prepared-dataset caches.
- [`data/staged/`](/home/marti/Code/PSDCodec/data/staged) and
  [`data/schemas/`](/home/marti/Code/PSDCodec/data/schemas) are currently empty.

Rule:
`data/raw/` is the correct home for untouched acquisitions. Prepared `.npz`
caches under `data/processed/` are reproducible outputs and should not be
versioned.

### `models/`

Trained and exported model artifacts.

- `models/checkpoints/` is the local output root for training checkpoints.
- `models/exports/` is the local output root for deployment-ready bundles.
- [`models/archive/`](/home/marti/Code/PSDCodec/models/archive) and
  [`models/benchmarks/`](/home/marti/Code/PSDCodec/models/benchmarks) are empty.

Rule:
these files are correctly allocated inside `models/`, but they are reproducible
outputs and should not be versioned.

### `reports/`

Reserved location for generated evaluation outputs.

The directory exists with the intended subfolders:

- [`reports/figs/`](/home/marti/Code/PSDCodec/reports/figs)
- [`reports/benchmarks/`](/home/marti/Code/PSDCodec/reports/benchmarks)
- [`reports/exports/`](/home/marti/Code/PSDCodec/reports/exports)
- [`reports/archive/`](/home/marti/Code/PSDCodec/reports/archive)

All are currently empty.

Rule:
if future work generates benchmark tables, publication plots, or presentation
bundles, they should land here instead of being mixed into `docs/` or `models/`.

### `notebooks/`

Interactive orchestration and presentation.

- [`notebooks/demo_deploy.ipynb`](/home/marti/Code/PSDCodec/notebooks/demo_deploy.ipynb)
  is the single canonical notebook.

Rule:
the notebook should stay thin and defer reusable logic to `src/interfaces/` and
other source modules.

## Current Allocation Assessment

### Well Allocated

These areas are structurally strong and aligned with the theory:

- `src/codec`, `src/objectives`, and `src/pipelines` reflect the manuscript well.
- `src/interfaces/deployment.py` correctly exposes the explicit deployment
  boundary instead of burying it in the notebook.
- `configs/experiments/demo.yaml`, `scripts/jobs/train_demo.py`, and
  `notebooks/demo_deploy.ipynb` form a coherent demo path.
- `tests/specs` and `tests/integration` cover the main component and workflow risks.

### Acceptable Placeholders

These directories are empty but not problematic by themselves:

- `configs/ci`, `configs/docker`, `configs/envs`, `configs/runtime`
- `docs/figs`, `docs/progress`
- `data/staged`, `data/schemas`
- `models/archive`, `models/benchmarks`
- `reports/*`
- `scripts/generate`, `scripts/release`, `scripts/tools`
- `tests/benchmarks`, `tests/e2e`
- `src/infra`

They become a problem only if they stay empty while contributors start placing
unrelated files elsewhere instead of using the intended boundary.

### Needs Cleanup or Policy Clarification

These locations deserve explicit attention before adding major features:

- Local generated metadata under `*.egg-info/` should stay out of source control.
- Generated caches under [`data/processed/`](/home/marti/Code/PSDCodec/data/processed)
  should stay out of source control.
- Generated checkpoints and export bundles under
  [`models/checkpoints/`](/home/marti/Code/PSDCodec/models/checkpoints) and
  [`models/exports/`](/home/marti/Code/PSDCodec/models/exports) should stay out
  of source control.
## Placement Rules Going Forward

When adding new material:

- Put manuscript and design prose in `docs/`.
- Put reusable implementation in `src/`.
- Put raw acquisitions in `data/raw/`.
- Put processed reusable datasets in `data/processed/` only if the repository
  explicitly intends to version them.
- Put trained and exported model artifacts in `models/`.
- Put generated figures and benchmark outputs in `reports/`.
- Keep notebooks orchestration-only and move reusable code out of them.
- Avoid adding more generated metadata inside `src/`.

## Practical Conclusion

The repository structure is mostly disciplined and already supports serious
iteration. The highest-value cleanup is not a large reorganization; it is a small
policy tightening:

1. keep root documentation present and authoritative,
2. keep generated demo artifacts reproducible but unversioned,
3. keep generated packaging metadata out of `src/` version control,
4. keep future outputs flowing into `reports/` instead of accumulating ad hoc.
