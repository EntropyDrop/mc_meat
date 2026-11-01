# MC Meat

Classify each visible texel of a 64x64 Minecraft UV skin as **meat** or **non-meat**. Input is the UV texture alone; a source photograph is not required. Inner and outer layers are classified independently.

- **Meat:** exposed skin, eyes (including whites and pupils), lips and mouth, and exposed hands, feet and body.
- **Non-meat:** hair, eyebrows, beard, glasses, headwear, clothing and accessories.
- **Ignore:** transparent or invalid atlas cells and inner cells completely covered by the enlarged outer layer in a face-normal orthographic projection.

The current research pipeline combines frozen DINOv2 features, local UV texture, physical surface connectivity and a facial decoder. The DDJ annotation UI is maintained separately in the dataset repository's `best_annotator/` directory. This repository contains code, schemas, tests and documentation. Training data, dependencies, weights, feature caches and deployment state are not distributed with it.

## Local configuration

Use a Python environment with PyTorch, torchvision, NumPy and Pillow. The early teacher probes also require transformers, OpenCV, ftfy and regex. The supervised experiments used Python 3.12 and PyTorch 2.8.0; exact dependency and input provenance belongs to each local run.

Provide the DINOv2 source checkout and ViT-L/14-registers weights, the differentiable Minecraft renderer and the matching view mapping assets. These dependencies are not automatically downloaded by training. The six views are `front_left`, `back_left`, `front_right`, `back_right`, `head_above_front` and `head_above_back`.

Copy `local_paths.example.json` to `local_paths.json`, edit its paths and select it explicitly:

```sh
cp local_paths.example.json local_paths.json
export MC_MEAT_CONFIG="$PWD/local_paths.json"
```

Alternatively, store the configuration at `~/.config/mc_meat/paths.json`. Relative JSON paths resolve from the configuration file's directory. Environment variables override individual settings: `MC_MEAT_DINO_SOURCE`, `MC_MEAT_BACKBONE`, `MC_MEAT_RENDERER`, `MC_MEAT_ASSETS`, `MC_MEAT_PROBE_LAYOUT` and `MC_MEAT_PROBE_QUEUE`. The final two are needed only for the early teacher probes. An explicitly selected missing or malformed configuration is an error.

Keep actual machine addresses, usernames, credentials and private paths in local configuration. `local_paths.json`, `.env*`, `.models/`, `data/`, `cache/`, `runs/`, `jobs/`, `service_runtime/`, `diagnostics/` and `experiments/` are ignored. Only the sanitized example belongs in Git.

The optional teacher downloaders respect standard `HTTP_PROXY`, `HTTPS_PROXY` and `NO_PROXY` environment settings; no proxy address is embedded in their source.

## Annotation and visibility

`uv_visibility.py` computes full same-part, face-normal orthographic coverage using exact rational intervals. The outer shell's total dimension increase is 1 for the head and 0.5 for torso/limbs. An inner cell is ignored only when every positive-area overlap is covered by fully opaque outer cells. Partial exposure, partial alpha and pose-dependent occlusion by another body part do not exclude it. RGB never determines visibility.

Binary masks use `mc-meat-v2`: 64x64 RGBA, white for meat, black for non-meat and transparent for ignore. Training preserves the original skin, a visibility-filtered skin, the target and its valid mask. Feature extraction, loss, metrics and predictions exclude ignored cells. Ignored input RGBA is cleared to zero.

Current optional fine classes are mutually exclusive:

| PNG class ID | Class | Binary target |
| --- | --- | --- |
| 1 | Exposed skin | Meat |
| 2 | Eyes | Meat |
| 3 | Mouth | Meat |
| 5 | Eyebrows | Non-meat |
| 6 | Beard | Non-meat |

ID 0 means unknown fine class and does not imply non-meat. Such cells still participate in binary training when a binary label exists. Hair, glasses and other clothing/accessories are not fine categories; retired IDs 4, 7 and 8 are rejected by the current schema. Fine labels must agree with saved binary labels and skin/visibility hashes. Saving labels does not trigger training.

The classifier currently validates classic arms and binary alpha only. Visibility computation itself also supports slim geometry and never treats partially transparent outer cells as complete coverage. Before retraining legacy labels, use the DDJ visibility migration tool in dry-run mode, inspect its report, then apply it. Annotation and classifier visibility implementations must agree.

## Models and supervision

`meat_features.py` supplies local, face-isolated 3x3 RGBA features and frozen multiview DINO features. `face_features_v3.py` adds isolated face-normal DINO observations for every valid UV cell, including bottom faces. Features are recovered through UV correspondence checks; image background is not a feature source.

`surface_model_v4.py` connects physically adjacent cells within the same part and layer instead of mixing unrelated neighbors in the UV atlas. It uses deterministic segment reductions, a binary head and five auxiliary logits. Earlier eight-class modules remain for historical checkpoint compatibility.

`face_decoder_v5.py` adds masked 8x8 inner/outer front-face convolutions, per-skin attention and head-face context. `face_decoder_v5b.py` restricts fine supervision to the head front. DINO stays frozen. Human fine edits receive full weight; unreviewed assistant proposals retain their recorded confidence weights. Labels never enter inference.

`train_face_v6.py` compares the original classifier retrained on the latest labels with the facial decoder with and without fine supervision. Splits hold out whole skin/duplicate groups. A fold must never initialize from weights trained on its held-out skin. Final all-data models are saved separately from out-of-fold predictions.

## Latest recorded comparison

The `face_v6_latest_20261101` experiment used 34 groups, 200 fixed epochs, seed 42 and a fixed binary threshold of 0.5.

| Model | Macro meat IoU | Front-face pooled IoU |
| --- | ---: | ---: |
| Original classifier, retrained | 70.94% | 89.98% |
| Facial decoder, binary only | 77.50% | 89.51% |
| Facial decoder, auxiliary fine labels | 77.75% | 92.01% |

Auxiliary supervision improved eyebrow exclusion but still missed eyes. The snapshot contained no positive beard labels and only eight mouth cells. These are historical out-of-fold estimates, not a new blind test or proof of stable overall auxiliary gains. Running an experiment does not deploy its model.

Example commands, after preparing a validated snapshot and compatible feature cache:

```sh
python train_face_v6.py --data data/face_v6_latest_20261101 --cache cache/semantic_v3_20261101 --output runs/NEW_RUN
python report_face_v6.py --run runs/NEW_RUN
python audit_face_v6.py --run runs/NEW_RUN
python predict_face_v5.py --checkpoint runs/NEW_RUN/face_front_fine/model.pt --input skin.png --output runs/prediction/meat.png --cache cache/semantic_v3_20261101
```

The comparison driver also requires its referenced historical OOF runs and a validated annotation audit. Adapt those experiment references explicitly for a new study. This is a research workflow, not a bundled dataset or pretrained installation. The prediction endpoint currently returns a binary mask; the five-class head provides auxiliary supervision.

## Checkpoint and cache compatibility

Feature/code hashes are enforced. A source or configuration change requires a new cache/run; never edit frozen provenance to make a mismatch pass. Existing models retain the implementation copied to their run's `source/` directory. For an older checkpoint, invoke its archived inference script, for example:

```sh
python runs/EXISTING_RUN/source/predict_face_v5.py --checkpoint runs/EXISTING_RUN/face_front_fine/model.pt --input skin.png --output runs/new_prediction/meat.png --cache cache/EXISTING_CACHE
```

The script name depends on the model family. Preserve that run's original dependencies and paths. New frozen source snapshots include `local_paths.py`; their private dependency configuration must be supplied locally. The annotation broker verifies legacy source hashes before delegating to archived inference.

## Manual annotation model service

`model_service.py` is an SSH/JSON broker for the DDJ annotator. It exposes no HTTP listener. Its `status` operation does not load CUDA. Prediction and training are detached jobs started only by explicit UI actions. Active state lives in `service_runtime/production/active.json`; a prediction pins its model at submission.

A manual update freezes saved skin/mask/metadata pairs, rejects stale or unsupported labels, groups near duplicates and reuses only provenance-matched features. The current update pipeline compares all-face and surface models with available five-class labels, verifies saved/reloaded inference and promotes a model only after successful checks. Historical MLP training uses CPU; feature extraction and newer surface training use short-lived GPU workers. Repeated update clicks attach to the active job. Failures preserve the previous model. Experimental facial decoder runs do not automatically replace production.

Inference does not modify DDJ files or turn predictions into human labels. Uncommitted, explicitly saved working-tree labels can be snapshotted. Jobs retain code, inputs, hashes, logs and prior model identity locally.

## Historical experiments and tests

- [Supervised v1](SUPERVISED_README.txt): six-skin baseline and reproduction details.
- [Supervised v2](SUPERVISED_V2_README.txt): thirteen-skin follow-up.
- [Label-free feasibility](README.txt): DINO/text and clothing-parser probes; coverage is not accuracy.

```sh
python -m unittest -v test_pseudo_gates test_supervised test_visibility test_surface_v3 test_semantic_categories_v2
```

External models and dependencies retain their own licenses. See `LICENSE` for this repository's license.
