MC Meat - supervised follow-up v2 (2026-10-29)

Historical record. See README.md for current configuration, visibility handling,
five-class auxiliary labels and the frozen-DINO facial decoder.

Dataset revision: 4b013c5421e1d50e865475218a9f7bba0c0b6abf.
Thirteen human-annotated skins contained 3,215 meat cells. Each whole-skin
leave-one-out fold trained on twelve skins. DINOv2 remained frozen; the local/
semantic fusion head trained for 200 fixed epochs with threshold 0.5.

Macro meat IoU across all thirteen skins:
  Local texture:       64.00%
  DINO + local texture: 70.26%
For the same six cases present in v1, DINO + local IoU rose from 50.49% to 67.18%.
These are historical leave-one-out estimates, not a fresh blind test. Errors
remain; model proposals should not be automatically accepted as human labels.

Local artifacts relative to the repository:
  data/human_v2_20261029
  runs/human_v2_20261029/dino_local/model.pt
  runs/human_v2_20261029/review_report_portable.html
  runs/human_v2_20261029/progress_report_portable.html

Inference for a compatible implementation:
python predict_meat.py --checkpoint runs/human_v2_20261029/dino_local/model.pt --input skin.png --output runs/my_prediction/skin_meat.png

For an older checkpoint whose feature hashes differ, use its archived
source/predict_meat.py. Do not bypass provenance checks. Only 64x64 classic
skins with binary alpha are supported. Use a new output path and preserve
human annotations. Save/reload and independent inference on additional cases
passed. See SUPERVISED_README.txt for the original pipeline details.

Annotation interaction introduced with this follow-up:
- Drag to rotate the 3D preview; click to toggle a cell.
- Ctrl-click to flood-select similar connected colors; repeat on a selected
  region to deselect it.
- Color tolerance ranges from 0 to 100%, default 12%.
- Connectivity follows the same part and layer across face seams, without
  crossing transparent cells.

The annotation UI belongs to the separate DDJ dataset repository. Its private
host address and deployment paths are configured locally, not published here.
