"""Mutually exclusive five-category suggestions, consistent with binary meat.

The auxiliary head is not an open-set segmenter: it has no hair/clothing class.
Keep binary negatives unselected unless a confident facial non-meat category is
predicted. Low-confidence binary positives remain bare-skin suggestions. These
are editable model suggestions, never automatically saved annotations.
"""
import numpy as np
from semantic_schema import CLASS_IDS, SCHEMA

def decode_categories(probability, fine_probability, valid, fine_scope, threshold=.5, fine_threshold=.7, class_support=None):
    p=np.asarray(probability);valid=np.asarray(valid,dtype=bool);scope=np.asarray(fine_scope,dtype=bool)
    assert p.shape==valid.shape==scope.shape==(4096,)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    positive=valid&(p>=threshold);labels=np.zeros(4096,np.uint8);labels[positive]=1
    available=fine_probability is not None
    if available:
        fine=np.asarray(fine_probability)
        assert fine.shape==(4096,len(CLASS_IDS)) and np.isfinite(fine).all()
        assert ((fine>=0)&(fine<=1)).all()
        assert np.allclose(fine[valid].sum(1),1,atol=1e-5)
        winner=np.asarray(CLASS_IDS)[fine.argmax(1)]
        confident=valid&scope&(fine.max(1)>=fine_threshold)
        if class_support is not None:
            allowed=np.asarray(class_support,dtype=bool);assert allowed.shape==fine.shape
            confident&=allowed[np.arange(4096),fine.argmax(1)]
        # Eyes/mouth never turn a binary non-meat cell into meat; brows/beards
        # never turn a binary meat cell into non-meat.
        keep=confident&((positive&np.isin(winner,[1,2,3]))|(~positive&np.isin(winner,[5,6])))
        labels[keep]=winner[keep]
    selected=np.flatnonzero(positive).tolist()
    assert selected==np.flatnonzero(np.isin(labels,[1,2,3])).tolist()
    return dict(labels=labels.tolist(),selected=selected,label_schema=SCHEMA,
                semantic_available=available,class_ids=CLASS_IDS,
                category_counts={str(c):int((labels==c).sum()) for c in CLASS_IDS},
                threshold=threshold,fine_threshold=fine_threshold if available else None)
