MC Meat - label-free feasibility experiment (2026-10-29)

Historical record. Supervised training subsequently replaced this prototype.
See README.md for the current five-class auxiliary decoder and configuration.
The six-skin supervised baseline reached 50.49% macro IoU; the thirteen-skin
follow-up reached 70.26%. Neither result is a new blind benchmark.

Task
Input: a 64x64 RGBA Minecraft UV skin. Classify valid cells as meat/non-meat.
Meat includes exposed skin, eyes, lips/mouth and exposed body parts. Hair,
eyebrows, beard, glasses, hats, crowns, clothing and accessories are non-meat.
Either class may occur in either layer. Invalid/transparent cells are ignored.
This early probe stores unknown candidates as 255; do not confuse that
internal representation with supervised binary mask encodings.
Only classic skins with binary alpha were validated.

Findings
1. Ordinary DINOv2 clustering captures similarity, not the project's meat
   definition. Cluster colors have no intrinsic class meaning.
2. dino.txt supplied pretrained text/image semantics but still selected parts
   of hoodies, coats and trousers as meat. Across eight cases, 4,077 of 15,458
   valid opaque cells were accepted (26.37% coverage); 2,981 were meat
   candidates. These numbers describe coverage, not accuracy.
3. The FASHN clothing/body parser supplied some clothing negatives but often
   ignored the head or misclassified exposed torso regions. Errors remained
   after using its official OpenCV INTER_AREA preprocessing.
4. Requiring both teachers to pass their filters and agree left 68 cells
   (0.44%), all non-meat. Zero meat positives cannot train a useful binary
   classifier. This batch should not be expanded as trusted pseudo-labels.
5. This was a qualitative probe of eight training-pool skins without human
   ground truth. No IoU, accuracy, precision or recall was established.
   It does not prove that all automatic-labeling approaches fail.

Artifacts
  download_dinotxt.py   Download official dino.txt weights and record provenance.
  download_parser.py    Download FASHN weights at a fixed HF revision.
  probe_no_labels.py    Six views, combined/inner/outer layers, head crops,
                       DINO clustering and dino.txt candidates.
  probe_parser.py       Clothing-parser teacher and agreement filtering.
  test_pseudo_gates.py  Reject conflicts between output class and voting support.
  build_review.py      Build an offline visual report from experiment outputs.
  task_schema.json     Classification boundaries.
  training_strategy.json  Historical proposed next steps, not completion evidence.

Weights, candidate masks, evidence and source snapshots remain in ignored
.models/ and runs/ directories. No meat classifier was trained by this probe.

Reproduction
Run from the repository root in an environment with the dependencies listed
in README.md. Configure DINO, renderer assets, probe_layout and probe_queue
using local_paths.json or environment variables. probe_layout must contain
the original layout.py; probe_queue is a train-split JSONL of skin records.
The probe reads skins and split metadata, not human labels.
Output directories must be new to preserve prior experiments.

python -m unittest -v test_pseudo_gates
python probe_no_labels.py --output runs/my_dino_probe --limit 8
python probe_parser.py --source runs/my_dino_probe --output runs/my_teacher_compare
python build_review.py --source runs/my_dino_probe --teacher runs/my_teacher_compare --output runs/my_review

Possible follow-up work
- Establish reliable automatic labels before fitting a student. Observe full
  bodies and enlarged heads separately for combined, inner and outer layers.
- Text-driven segmentation and local visual-language checks remain candidate
  teachers, not validated solutions. Self-reported confidence is not accuracy.
- Synthetic truth is valid only when generation records the actual semantic
  materials/parts. Alpha, layer, skin-color thresholds and coordinates are not
  semantic labels. Synthetic style must also match the target skin library.
- Verify exact UV ID/depth/visibility projection and exclude background.
  Preserve uncertain or conflicting cells as unknown. Cover bottom and other
  surfaces not observed by the initial six views.
- A student can combine frozen multiview features with a detailed UV branch,
  joining physical faces rather than unrelated neighbors in the atlas.
- Use reliable labels for supervised loss; consistency is auxiliary. Repeating
  wrong pseudo-labels or lowering confidence thresholds does not repair truth.
- Split by character/duplicate family. Without human truth, report coverage,
  conflict rates, augmentation consistency and synthetic metrics, not accuracy
  on the real skin library.

Version notes
v1 retained the original voting filter and HF automatic preprocessing.
v2 fixed class-supported voting and used official INTER_AREA preprocessing.
The conclusions above use v2. Agreement 0.8, dino.txt margin 0.015, parser
probability 0.7 and at least two views were uncalibrated screening gates,
not production classifier thresholds or UV outer-layer thresholds.

Sources (each retains its original license)
DINOv2: https://github.com/facebookresearch/dinov2
DINOv2 Meets Text: https://arxiv.org/abs/2412.16334
FASHN: https://huggingface.co/fashn-ai/fashn-human-parser
Parser preprocessing: https://github.com/fashn-AI/fashn-human-parser/blob/main/src/fashn_human_parser/parser.py
Candidate text segmentation: https://github.com/facebookresearch/sam3
