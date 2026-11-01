MC Meat - supervised baseline v1 (2026-10-29)

Historical record; see README.md for the current visibility policy, five-class
auxiliary supervision, facial decoder and portable dependency configuration.

Data and method
Six complete human annotations were frozen from DDJ Git/LFS revision
e7dc0a706a44dd33abe1305b257f1e14c9465886, excluding annotation history,
unlabeled skins and automatic pseudo-labels. There were 10,438 valid cells,
including 1,160 meat cells. The independent sample count was six skins.

The model takes only 64x64 RGBA UV textures. Only classic geometry and binary
alpha were validated. Offline DINOv2 ViT-L/14-registers stays frozen. Features
combine six views of combined/inner/outer layers, head crops and local 3x3
face-isolated RGBA texture. UV-ID and color checks exclude background.
Unobserved cells retain their local texture features. A fixed random
projection compresses DINO features to 128 dimensions without fitting data.
Part, face, layer, coordinates and face color statistics provide context,
not class rules. Unrelated atlas faces do not share local neighborhoods.

A shared MLP trains for 200 epochs with AdamW, learning rate 0.002 to 0.0002
and weight decay 0.01. BCE weights skins equally; each training fold uses
sqrt(negative/positive) positive weighting clipped to [1,5]. Only valid
opaque cells contribute. A complete all-negative annotation is legitimate;
a missing annotation is not. Predictions never modify the input texture.

Legacy mask format
This experiment used 64x64 grayscale PNGs: 255=meat, 0=non-meat on valid
cells. Invalid/background cells were also zero and excluded using skin alpha
and atlas layout. Current mc-meat-v2 masks instead use RGBA with transparent
ignore cells. Migrate legacy annotations before training the current code.
Do not confuse either format with early pseudo-label candidates (255=ignore).

Whole-skin leave-one-out validation
Each fold trained on five skins and held out the sixth. No random pixel split.

                          Macro IoU  Macro precision  Macro recall  Macro Dice
Local texture                44.94%           61.25%        80.47%      57.12%
DINO + local texture         50.49%           63.64%        83.67%      63.09%

Macro metrics give each skin equal weight. Pooled DINO metrics were IoU
39.83%, precision 49.19% and recall 67.67%. Both variants used fixed epochs
and threshold 0.5; held-out labels did not select epochs or thresholds.
The near-duplicate check found six groups but cannot rule out all related
character families. These are historical samples, not a new blind test.
Six skins do not establish generalization. Final models use all six skins;
reports show OOF predictions. Each fold checkpoint records its training IDs.

Main failures included missed bare arms, clothing/armor mistaken for meat
and errors around cuffs and heads. Prioritize diverse sleeves, bare arms,
gloves, skin-colored clothes, facial hair, glasses, skin tones and non-human
characters. More epochs alone are unlikely to solve coverage gaps. Predictions
require human review before becoming annotation truth.

Local artifacts, relative to the repository
  data/human_v1_20261029
  cache/human_v1_20261029
  runs/human_v1_20261029/dino_local/model.pt
  runs/human_v1_20261029/local/model.pt
  runs/human_v1_20261029/review_report_portable.html
  jobs/human_v1_20261029/pipeline.log

Inference and reproduction
Run from the repository root after configuring dependencies as in README.md.
For existing checkpoints, use the corresponding archived source/predict_meat.py
and preserve original provenance. The following commands illustrate the current
entry points for a compatible checkpoint or a newly prepared run:

python predict_meat.py --checkpoint runs/human_v1_20261029/dino_local/model.pt --input skin.png --output runs/my_inference/skin_meat.png
python -m unittest -v test_supervised
python prepare_human.py --dataset ../SKING_DDJ_Dataset --revision COMMIT_SHA --output data/NEW_RUN
python train_supervised.py --data data/NEW_RUN --cache cache/NEW_RUN --output runs/NEW_RUN

Output paths must be unused. Inference writes a mask, floating probabilities
and an audit. Do not write predictions over human DDJ masks. DINO uses GPU by
default and exits after the process; --device cpu is available but slower.
The local-only model does not need DINO. Seen-input checks use hashes, not names.

Preparation verifies each saved annotation's completeness, geometry and skin/
mask hashes. Feature caches verify input, backbone, renderer, mapping and code
hashes. Feature extraction does not read labels. A feature implementation change
requires new caches. Preserve frozen runs/source and existing provenance.

As the sample count grows, use fixed character/family train/validation/test
splits or grouped K-fold evaluation instead of increasingly costly leave-one-out.
Save/reload and independent held-out UV inference were checked for this run.
Evidence is in comparison.json, inference_audit.json, verification.json and
release.json. Weights, training data and caches are local, ignored artifacts.
