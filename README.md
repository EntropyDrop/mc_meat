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

The original five-class checkpoints use these mutually exclusive fine classes:

| PNG class ID | Class | Binary target |
| --- | --- | --- |
| 1 | Exposed skin | Meat |
| 2 | Eyes | Meat |
| 3 | Mouth | Meat |
| 5 | Eyebrows | Non-meat |
| 6 | Beard | Non-meat |

ID 0 means unknown fine class and does not imply non-meat. Such cells still participate in binary training when a binary label exists. Hair, glasses and other clothing/accessories are not fine categories; retired IDs 4, 7 and 8 are rejected by the current schema. Fine labels must agree with saved binary labels and skin/visibility hashes. Saving labels does not trigger training.

The classifier currently validates classic arms and binary alpha only. Visibility computation itself also supports slim geometry and never treats partially transparent outer cells as complete coverage. Before retraining legacy labels, use the DDJ visibility migration tool in dry-run mode, inspect its report, then apply it. Annotation and classifier visibility implementations must agree.

## Seven-class reviewed training

`semantic_schema_v3.py` extends the editable taxonomy without changing historical five-class checkpoint definitions. Class 9 is eyeshadow (explicit makeup, excluding eyeliner/lashes); class 10 is nose. Both belong to meat. Eyebrows and beard remain non-meat. Retired IDs 4, 7 and 8 remain invalid. Class 0 represents other only after explicit whole-skin review; partial unknown labels continue to receive binary supervision alone.

The `face_v9_seven_20261102` experiment freezes all 169 fully reviewed skins from dataset commit `3d00dad0`. It uses frozen DINO features, exact visibility filtering, and five whole-skin/near-duplicate grouped folds. Two jointly supervised seven-class models use identical splits, 200 epochs, random decoder initialization and a fixed 0.5 confidence threshold:

- `surface_v9`: surface adjacency, part/layer pooling and per-skin context.
- `face_v9`: the same backbone plus paired inner/outer convolutions and attention across all six head faces.

The internal softmax includes other plus the seven editable categories. Binary meat probability sums skin, eyes, mouth, eyeshadow and nose probabilities. Fine false positives are evaluated across all reviewed valid cells, including unselected other cells. Historical validation is not a fresh blind benchmark. Beard (5 skins), eyeshadow (8) and nose (9) have few positive examples, so per-class results need careful review.

`prepare_face_v9.py` reads committed Git/LFS objects, verifies PNG/JSON consistency and complete review metadata, then reuses features only when code, input and cache hashes match. `run_face_v9.py` produces same-fold comparisons, a separate historical production reference, full-data weights, standalone cached/uncached inference audits, and a portable report. It never changes the active annotation model. The existing manual training service remains on the historical five-class pipeline until a seven-class candidate is explicitly integrated and selected.

```sh
python prepare_face_v9.py --dataset /path/to/dataset --revision COMMIT --data data/NEW_RUN --cache cache/NEW_RUN_multiview --faces cache/NEW_RUN --job jobs/NEW_RUN
python run_face_v9.py --data data/NEW_RUN --cache cache/NEW_RUN --output runs/NEW_RUN --reference /path/to/current/surface/model.pt --job jobs/NEW_RUN
python predict_face_v9.py --checkpoint runs/NEW_RUN/face_v9/model.pt --input skin.png --output runs/new_prediction --cache cache/NEW_RUN
python test_face_v9.py
```

`loss_face_v9b.py` adds an isolated rare-class balancing experiment after a training-fit diagnostic found that eyeshadow and nose were not learned even on training examples. It averages cross-entropy within each present training class, then across classes, while keeping binary supervision unchanged. `train_face_v9b.py`, `run_face_v9b.py` and `report_face_v9b.py` use the same frozen data, folds, initialization, epochs and thresholds. Both original and balanced results are retained; this is an exploratory development comparison, not an independently selected blind test. Use `test_face_v9b.py` to verify equal per-class gradient mass and partial-label handling.

For reproducibility, run completed checkpoints using the scripts archived under their run's `source/` directory. Supply dependency paths through local configuration. Prediction consumes only the skin and optional hash-matched image features; labels are never inference inputs.

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

The comparison driver also requires its referenced historical OOF runs and a validated annotation audit. Adapt those experiment references explicitly for a new study. This is a research workflow, not a bundled dataset or pretrained installation.

## Checkpoint and cache compatibility

Feature/code hashes are enforced. A source or configuration change requires a new cache/run; never edit frozen provenance to make a mismatch pass. Existing models retain the implementation copied to their run's `source/` directory. For an older checkpoint, invoke its archived inference script, for example:

```sh
python runs/EXISTING_RUN/source/predict_face_v5.py --checkpoint runs/EXISTING_RUN/face_front_fine/model.pt --input skin.png --output runs/new_prediction/meat.png --cache cache/EXISTING_CACHE
```

The script name depends on the model family. Preserve that run's original dependencies and paths. New frozen source snapshots include `local_paths.py`; their private dependency configuration must be supplied locally. The annotation broker verifies legacy source hashes before delegating to archived inference.

## Manual annotation model service

`model_service.py` is an SSH/JSON broker for the DDJ annotator. It exposes no HTTP listener. Its `status` operation does not load CUDA. Prediction and training are detached jobs started only by explicit UI actions. Active state lives in `service_runtime/production/active.json`; a prediction pins its model at submission.

A manual update freezes saved skin/mask/metadata pairs, rejects stale or unsupported labels, groups near duplicates and reuses only provenance-matched features. `service_face_v8.py` now connects the manual button to the facial decoder, a same-split surface control, nested body confidence calibration, and training-only positive spatial support. It promotes the candidate only after inference, release-integrity and quality checks pass. Historical MLP training uses CPU; feature extraction and newer training use short-lived GPU workers. Repeated update clicks attach to the active job. Failed checks preserve the previous model and retain the candidate report.

Inference does not modify DDJ files or turn predictions into human labels. Uncommitted, explicitly saved working-tree labels can be snapshotted. Jobs retain code, inputs, hashes, logs and prior model identity locally.

### Five-category preselection

Compatible surface models return both the binary selection and 4,096 mutually exclusive category IDs: bare skin (1), eyes (2), mouth (3), eyebrows (5), beard (6), or unselected (0). DDJ applies all five categories as one undoable edit. Switching categories shows only that category's highlight; annotations are written only after an explicit save. Legacy binary models retain the previous behavior and report that fine categories are unavailable.

For legacy surface checkpoints, `semantic_preselection.py` preserves the binary meat decision. Fine suggestions require auxiliary confidence of at least 0.7 and training-supported positions on head faces. This heuristic cannot make a closed-set auxiliary classifier understand other categories. New facial checkpoints use the joint output described below.

`semantic_support.py` builds the position support from known fine labels in the checkpoint's own frozen training snapshot. It pools layers and expands each labelled position by one cell within its face. Unknown fine cells are never used as negative labels. Support is checkpoint-bound and stored under ignored `service_runtime/semantic_support`; it is prepared automatically by subsequent manual training. To enable an existing compatible checkpoint:

```sh
python semantic_support.py --checkpoint service_runtime/production/jobs/JOB/run/surface/model.pt --data service_runtime/production/jobs/JOB/data --cache service_runtime/production/jobs/JOB/face_cache
```

This remains a preselection aid: unusual facial layouts and sparse mouth/beard examples can be missed or misclassified. Historical front-only facial checkpoints retain their original supervised front scope.

### Joint facial decoder and review status

`face_decoder_v8.py` uses masked inner/outer grids for all six head faces, facial attention, surface neighbors and per-skin context. DINO remains frozen. Internally, the decoder predicts other, bare skin, eyes, mouth, eyebrows and beard; the annotator still exposes only the original five mutually exclusive categories. Unknown fine cells receive only binary marginal supervision, never an invented other label. Fine labels use class balancing and their recorded provenance weights.

The explicit complete-review checkbox records `fine_review_complete`. Only after the user confirms the whole valid skin may unselected cells supply other-category fine supervision. Old partial annotations retain their original status. The dataset fingerprint includes semantic metadata, so review-only changes trigger an update indication.

`calibrate_face_v8.py` uses three inner grouped folds inside each outer training fold to select body thresholds for a predeclared F0.5 objective. Outer labels never select their thresholds. Deployment thresholds come from whole-data OOF predictions. The precision mode reduces false selections at the cost of missed selections; balanced mode uses the same weights without body calibration. Neither changes the five UI categories. Uncertain cells remain unselected and are explicitly reported for review.

`train_face_v8.py` compares new and old structures on the same five grouped folds and saves final all-data weights separately. Current-model reference predictions use historical held-out checkpoints, or the old final model only for cases it never trained on. Fine metrics describe known labels with mixed human/assistant provenance, not a new blind benchmark or full-skin fine-class accuracy. A quality gate failure prevents automatic promotion.

```sh
python train_face_v8.py --data data/SNAPSHOT --cache cache/FACES --output runs/BASE_RUN --reference runs/PREVIOUS/face_v8/model.pt
python calibrate_face_v8.py --parent runs/BASE_RUN --data data/SNAPSHOT --cache cache/FACES --output runs/CALIBRATED_RUN
python rejection_face_v8.py --parent runs/CALIBRATED_RUN --output runs/CANDIDATE_RUN
python report_face_v8.py --run runs/CANDIDATE_RUN --cache cache/FACES
python -m unittest test_face_v8
```

`rejection_face_v8.py` abstains from eye, mouth, eyebrow and beard predictions outside the positions supported by known positive training labels, with a one-cell margin. Each outer fold constructs this support only from its own training partition. Both inference modes use it. This is an abstention aid, not learned hair/clothing recognition; unusual layouts may be missed. Unknown fine labels remain unknown. The mechanism was added after inspecting development predictions, so historical results are exploratory rather than a fresh blind evaluation.

## Historical experiments and tests

- [Supervised v1](SUPERVISED_README.txt): six-skin baseline and reproduction details.
- [Supervised v2](SUPERVISED_V2_README.txt): thirteen-skin follow-up.
- [Label-free feasibility](README.txt): DINO/text and clothing-parser probes; coverage is not accuracy.

```sh
python -m unittest -v test_pseudo_gates test_supervised test_visibility test_surface_v3 test_semantic_categories_v2
```

External models and dependencies retain their own licenses. See `LICENSE` for this repository's license.


## Rare-category specialists and nested calibration

The face_v10_rare_20261102 and face_v11_rare_20261102 experiments retain the ordinary seven-class model as the default prediction and train separate beard, eyeshadow and nose heads. Their supervision uses equal positive-skin means, the hardest 16 reviewed negative cells per skin/class, an all-negative term and an image-level presence task. Unknown fine cells and invisible cells do not become negative examples.

V10 inherits the ordinary decoder's frozen feature backbone. V11 tests a balanced decoder initialization for the specialists, retains the ordinary fallback, uses a bounded presence factor of 0.5 + 0.5 * presence_probability, and searches all positive inner-OOF score breakpoints instead of a fixed coarse grid. Both preserve seven mutually exclusive user-facing categories. Specialists may correct the fallback binary assignment when accepted; no hard skin-color rules or mandatory symmetry are introduced.

All comparisons retain the existing five outer skin/duplicate-group folds. Each outer training partition has three inner folds for threshold calibration. Every inner feature decoder and specialist excludes its own inner-held examples and the outer-held examples. V11 reuses only provenance-matched ordinary inner checkpoints and trains separate balanced inner initializers. Final full-data thresholds come from OOF predictions; they never replace the per-outer-fold thresholds in reported validation results.

Run the standalone predict_face_v10.py or predict_face_v11.py archived in each completed run's inference_source/ package with --checkpoint, --input, --output and optional --cache. The input is UV only. Training, inference, label/source hashes and cached-versus-fresh inference are audited separately. These experiments do not modify the active annotation model or overwrite human annotations.

These are exploratory historical comparisons, not a new blind test. There are only five beard-positive skins, eight eyeshadow-positive skins and nine nose-positive skins. Improved rejection can increase misses; always examine class-wise precision/recall and the portable report before selecting a model.


## Guarded rare-category calibration

The guarded derivatives preserve all trained tensors and original experiments. A specialist can override the ordinary decoder only if its inner-fold calibration has at least one true positive and precision of at least 0.35. Otherwise that category retains the ordinary result. This replaces the previous fallback that could select a very low threshold despite poor precision. Final full-data thresholds are separate from outer-fold evaluation thresholds.

Run the archived predict_rare_guard.py from a guarded run with --checkpoint, --input, --output and optional --cache. The checkpoint uses schema mc-meat-rare-guarded-v1 and records its parent model hash and per-class enabled flags. It accepts a UV image only. Disabled categories are not erased from the ordinary prediction, and category assignments remain mutually exclusive.

These derivatives are calibration changes, not additional retraining. They retain the fixed-threshold and unrestricted-calibration experiments in their reports. Rejecting more predictions is not evidence of improved recall; beard and nose remain limited by sparse positive skins. No experimental weights are automatically activated.


## Weak-label expansion with matched controls

`train_face_v12.py` compares ordinary and class-balanced face decoders, each trained with human labels alone or with additional weak annotations. All four arms use the same frozen DINO features, architecture, seed, optimizer schedule, 200 epochs and fixed 0.5 output confidence. The original held-out skin IDs are retained. Known same-source counterparts are removed from the training portion in both the control and augmented arms.

Assistant binary labels use their saved sample weight (0.5 for this batch). Fine supervision preserves per-cell confidence, and unknown fine categories receive no auxiliary loss. Complete human review permits supervision of the internal other class. A known assistant case overlapping historical data is excluded from augmentation. New weak examples are never evaluation targets or threshold-tuning examples.

The full-data checkpoints use the existing `mc-meat-face-v9` inference schema and record their training variant separately. Use the archived `inference_source/predict_face_v9.py` with a UV image, a selected checkpoint, an optional matching feature cache and a fresh output directory. Training and finalization do not activate the annotation service model.

```sh
python train_face_v12.py --data data/EXPANDED --cache cache/EXPANDED_FACES --output runs/EXPANSION --parent runs/ORDINARY_REFERENCE --balanced runs/BALANCED_REFERENCE --guarded runs/GUARDED_REFERENCE
python finalize_face_v12.py --run runs/EXPANSION --job jobs/EXPANSION --data data/EXPANDED --cache cache/EXPANDED_FACES
python -m unittest test_face_v12 test_face_v9
```

`rare_analysis.json` separates missing beard/eyeshadow/nose categories from incorrect binary meat assignments, reports false-positive origins and false-negative destinations, and includes exploratory paired intervals resampled by source group. The portable report can switch models, target category and error overlays. Unsupported categories in the old five-class online reference are shown as unavailable, not scored as a competing detector.

These remain historical development comparisons with few positive skins. Added assistant labels are weak supervision pending human review. Improvement on these folds is not proof of performance on a new blind dataset.


## Corrected labels and bounded head refinement

The v13 boundary experiments, v14 source-balanced/context experiments, and v15
corrected-label experiments are offline studies. They do not replace the active
annotation model. Frozen datasets, feature caches, checkpoints, reports and local
job records stay in ignored directories.

V13 preserves partial weak supervision and adds side/chin annotations, synthetic
upper-eyelid makeup and partially labeled hard negatives. Unknown cells do not
become fine-category negatives. V14 separates human and auxiliary supervision,
groups synthetic replicas by parent, and tests six-face head context, known hard
negatives and a learned presence gate. Its separately archived nested-calibration
experiment fits thresholds only on inner out-of-fold predictions from the outer
training partition. Strong rejection reduced false positives but lost true beards.
Use the `_r2` calibration and standalone inference entry points; earlier versions
are retained to reproduce the original frozen experiment and recovery provenance.

V15 uses a separate, user-confirmed correction of 88 beard cells in two historical
skins: white hair and hood/collar become other non-meat. Canonical annotations are
preserved. The portable annotation revision is recorded separately in the DDJ
dataset repository under `annotation_revisions/20261102_beard_correction`.
All historical predictions are re-scored against the same corrected targets.

Three matched v15 arms retain the same five source-disjoint folds, seed 42, frozen
DINO features, highest float32 matmul precision and fixed 0.5 output confidence:

- `original_labels_control_v15`: original labels and decoder/loss, 200 epochs.
- `corrected_labels_v15`: only the two confirmed label corrections, 200 epochs.
- `bounded_refine_v15`: initialize from the corresponding corrected-label fold,
  freeze the entire base decoder, and train six-face head refinement for 100 epochs.
  Each head logit residual is bounded to [-1, 1]; body logits remain unchanged.
  Fine supervision separates human and auxiliary means, gives the auxiliary term
  weight 0.5, and groups replicas by parent. Known hard negatives use weight 0.025
  and distribution preservation uses KL weight 0.05.

Never initialize a held-out fold from the full-data checkpoint. The trainer checks
fold membership and frozen weights; finalization audits all 18 checkpoints,
independent UV-only inference, cached versus fresh features and archived CLI use.

```sh
python train_face_v15.py --data data/CORRECTED --old-data data/ORIGINAL --cache cache/FACES --output runs/CORRECTED_RUN --parent runs/PARENT_COMPARISON
python finalize_face_v15.py --run runs/CORRECTED_RUN --job jobs/CORRECTED_RUN --data data/CORRECTED --cache cache/FACES
python runs/CORRECTED_RUN/source/predict_face_v15.py --checkpoint runs/CORRECTED_RUN/bounded_refine_v15/model.pt --input skin.png --output output/NEW_CASE
python -m unittest test_face_v13 test_face_v14 test_calibration_v14 test_face_v15
```

Inputs are UV images only. Semantic annotations are supervision or outputs, never
inference inputs. The finalizer expects the job's correction audit and preflight
record, and the trainer requires a fresh output directory. The archived source is
preferred when reproducing a frozen experiment.

On 169 historical development skins, the matched original-label control versus
bounded refinement produced beard FP/FN 2553/30 versus 843/50, eyeshadow 87/7 versus
77/7, and nose 125/9 versus 85/8. Beard precision improved but recall regressed; this
is not a complete solution or a deployment recommendation. Corrected beard support
is only 77 cells in three skins, including 72 cells in one character. Eyeshadow and
nose support remain 28 and 17 cells. These are reused development folds, not a new
blind test. Inspect per-case reports, missed positives and false-positive negatives
before considering any model change.
