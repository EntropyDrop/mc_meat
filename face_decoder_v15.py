"""Bounded head refinement of a frozen, fold-matched face decoder."""
import torch
from torch import nn
from face_decoder_v9 import FaceDecoder
from face_decoder_v14 import ContextDecoder,pack

class BoundedDecoder(ContextDecoder):
 def __init__(self,dim):
  super().__init__(dim,gated=False)
  self.expert=nn.Sequential(self.expert,nn.Tanh())
  self.base.requires_grad_(False);self.presence.requires_grad_(False);self.base.eval()
 def train(self,mode=True):
  super().train(mode);self.base.eval();return self

def build(dim,variant):
 if variant in ['original_labels_control_v15','corrected_labels_v15']:return FaceDecoder(dim)
 assert variant=='bounded_refine_v15';return BoundedDecoder(dim)
