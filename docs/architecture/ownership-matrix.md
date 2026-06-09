# Ownership Matrix

This is the concrete boundary matrix for the post-audit cleanup track.

Use it to answer three questions before moving code:

1. Which package is the canonical owner?
2. Which packages are compatibility-only?
3. Which follow-up track should touch the files?

## Package Matrix

| Package / Surface | Canonical Role | Compatibility Status | Allowed Import Direction | Follow-up Track |
| --- | --- | --- | --- | --- |
| `src.training` | Canonical trainer implementation layer | Canonical | `train.py`, `genlib`, `compat`, `pipelines.train` may depend on it | Track B / D |
| `src.sampling` | Canonical runtime sampler implementation layer | Canonical | `run_model.py`, `compat`, `pipelines.samplers` may depend on it | Track B / C |
| `src.pipelines.inference` | Canonical facade/orchestration layer | Canonical | May depend on `sampling`, `scheduling`, `models`, `utils` | Track D |
| `src.pipelines.samplers` | Legacy handler/delegation layer | Compatibility only | May depend on `src.sampling`, not vice versa | Track B / C |
| `src.pipelines.train.*_lib` | Legacy training wrappers | Compatibility only | May depend on `src.training`, not vice versa | Track B |
| `src.pipelines.utils` | Legacy re-export surface for scheduler/sampling helpers | Compatibility only | May depend on `src.scheduling`, not vice versa | Track B |
| `src.losses` | Trainer-facing loss components and `LOSS_REGISTRY` | Canonical | Trainers may depend on it | Track C |
| `src.nn.losses` | Tensor-level loss math and small adversarial primitives | Canonical primitive layer | `src.losses` may depend on it | Track C |
| `src.models` | Canonical model families + factory layer | Canonical | Trainers/samplers/facades may depend on it | Track D |
| `src.scheduling` | Canonical scheduler building, loop, conditioning, LR helpers | Canonical | Trainers/samplers/facades may depend on it | Track C / D |
| `src.utils.dataset_utils` | Dataset config resolution and dataset construction | Canonical | Trainers/samplers/facades may depend on it | Track C |
| `src.utils.dataset_runtime` | Dataset cache/output runtime helpers | Canonical | Datasets/samplers may depend on it | Track C |
| `src.utils.training_utils` | Config IO, checkpoint IO, generic training runtime helpers | Canonical | Trainers/root CLIs may depend on it | Track C |
| `src.utils.model_utils` | Model-family-specific helpers for build/encode/decode | Transitional | Facades/samplers/trainers may depend on it | Track C / D |
| `src.compat` | Migration and deprecation-preserving wrapper layer | Compatibility only | May depend on canonical layers, never the reverse | Track B |
| Root `train.py` / `run_model.py` | Supported user-facing CLIs | Canonical for now | Depend on canonical layers only | Track B |
| `src.train` | Deprecated compatibility CLI | Compatibility only | May depend on canonical training layer | Track B |

## Follow-up Track File Lists

### Track B: Compatibility Contraction

Primary files:

- `src/compat/legacy_training.py`
- `src/compat/legacy_samplers.py`
- `src/pipelines/train/vae_lib.py`
- `src/pipelines/train/diffusion_lib.py`
- `src/pipelines/train/flow_matching_lib.py`
- `src/pipelines/train/generative_lib.py`
- `src/pipelines/samplers/handlers/`
- `src/pipelines/samplers/abstract/`
- `src/pipelines/samplers/concrete/`
- `src/pipelines/utils.py`
- `src/train.py`
- docs that still show compatibility paths as normal entrypoints

Intent:

- hard-deprecate remaining non-canonical surfaces
- move wrappers under `src.compat` where practical
- reduce “two correct ways” documentation

Risk:

- medium
- mostly doc/import churn, low algorithm risk

### Track C: Module Decomposition

Primary files:

- `src/utils/training_utils.py`
- `src/utils/model_utils/diffusion_utils.py`
- `src/pipelines/samplers/diffusion_like.py`
- follow-on cleanup in `src/utils/dataset_utils.py` only if new seams are clear

Intent:

- split large modules after boundaries are frozen
- separate build/runtime/export responsibilities without changing ownership

Risk:

- medium to high
- import churn is real; behavior churn should be kept low

### Track D: Factory / API Decision

Primary files:

- `src/models/factory.py`
- `src/utils/model_utils/diffusion_utils.py`
- `src/utils/model_utils/vae_utils.py`
- `src/pipelines/inference.py`
- trainer `_build_model(...)` callsites
- docs/examples that teach model construction

Intent:

- decide whether `ModelFactory.build(...)` stays canonical
- if not, introduce a canonical `from_config()` style and migrate intentionally

Risk:

- high
- this changes the architectural story, not just file layout

Decision:

- `ModelFactory.build(...)` is the canonical model-construction API.
- model classes do **not** grow a parallel `from_config()` construction story.
- trainer/facade `from_config(...)` helpers remain valid because they own config IO and runtime orchestration.
- legacy generator wrappers (`models.generators.vaefactory`, `models.generators.diffusionfactory`) remain compatibility-only delegates over `ModelFactory.build(...)`.

### Track E: Final Repo-Wide Audit

Primary checks:

- full test suite
- `validate_refactor.sh`
- aggressive internal audit
- import-surface audit
- docs-path / compat-path audit

Intent:

- verify the cleanup did not create new drift

Risk:

- low implementation risk, high review-value

## Order of Operations

Recommended order:

1. Track B
2. Track D
3. Track C
4. Track E

Rationale:

- contract the legacy surface first
- settle the canonical build API before splitting modules around it
- decompose files only after ownership and public API are stable

## Stop Conditions

Track A is complete when:

- boundary docs define canonical ownership for the overlapping packages
- every non-canonical surface is explicitly labeled compatibility-only
- follow-up tracks have a concrete file inventory and order

That condition is what this page is intended to satisfy.
