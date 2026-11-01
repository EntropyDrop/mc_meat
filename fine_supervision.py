"""Fine supervision provenance and weights. Unknown cells have exactly zero loss."""
import math
import numpy as np
def weights(metadata):
 labels=metadata['labels']
 if metadata.get('human_saved') is True and metadata.get('annotation_source','human') in {'human','human_reviewed'}:
  return np.array([1. if x else 0. for x in labels],np.float32)
 source=metadata.get('annotation_source');assert source in {'assistant_visual_review_20261101','mixed_human_assistant'},'Unknown fine annotation provenance'
 assert metadata.get('human_saved') is (source=='mixed_human_assistant')
 human=metadata.get('human_reviewed_cells',[]);assert isinstance(human,list) and all(type(i) is int and 0<=i<4096 for i in human)
 assert not human or source=='mixed_human_assistant';human=set(human)
 w=metadata['fine_weights'];assert isinstance(w,list) and len(w)==4096
 assert all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=(1. if i in human else .8) and (labels[i]>0 or x==0) for i,x in enumerate(w))
 return np.array(w,np.float32)
