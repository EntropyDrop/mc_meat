"""Reject unsupported rare-class overrides using inner calibration evidence."""
import numpy as np

RARE_IDS = [6, 9, 10]
MIN_PRECISION = 0.35


def guarded_thresholds(records):
    assert [r['class_id'] for r in records] == RARE_IDS
    enabled = [r['tp'] > 0 and r['precision'] >= MIN_PRECISION for r in records]
    thresholds = [float(r['threshold']) if ok else 1.001 for r, ok in zip(records, enabled)]
    return thresholds, enabled


def decode(base, score, valid, thresholds, enabled):
    from face_decoder_v9 import MEAT_IDS
    labels = np.asarray(base, dtype=np.uint8).copy()
    valid = np.asarray(valid, dtype=bool)
    enabled = np.asarray(enabled, dtype=bool)
    assert score.shape == (len(labels), 3) and np.isfinite(score).all()
    assert len(thresholds) == len(enabled) == 3 and min(thresholds) > 0
    # A rejected specialist leaves the ordinary decoder's class intact.
    for cls, ok in zip(RARE_IDS, enabled):
        if ok:
            labels[labels == cls] = 0 if cls == 6 else 1
    ratio = np.where(enabled[None, :], score / np.asarray(thresholds)[None, :], -1.)
    take = valid & (ratio.max(1) >= 1)
    labels[take] = np.asarray(RARE_IDS, dtype=np.uint8)[ratio.argmax(1)[take]]
    labels[~valid] = 0
    return dict(labels=labels.tolist(), selected=np.flatnonzero(np.isin(labels, MEAT_IDS)).tolist(),
                uncertain=[], uncertain_count=0, semantic_available=True,
                class_ids=[1, 2, 3, 5, 6, 9, 10], label_schema='mc-meat-semantics-v3', human_label=False)
