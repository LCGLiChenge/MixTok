"""Official1D initialization; NEW2D/spatial/fusion parameters, no local ckpt."""
import torch
from torch import nn
from h20.assets import official_model
from h20_joint.spatial import SpatialTiTokSparseImageBert
from h20_joint.structure import OutputSpatialTiTokSparseImageBert
from h20_joint.fusion import AcceptanceGate


class JointSystem(nn.Module):
    def __init__(self,generator):
        super().__init__();self.generator=generator;self.fusion=AcceptanceGate('zero')


def create(root,memory='local'):
    if memory not in ('local','full'):raise ValueError('choose local or full spatial memory')
    original,report=official_model(root)
    kwargs=dict(attention_implementation=original.attention_implementation)
    generator=(SpatialTiTokSparseImageBert(**kwargs) if memory=='local' else
               OutputSpatialTiTokSparseImageBert(enhancement_kind='output_cross',memory_scope='full',**kwargs))
    result=generator.load_state_dict(original.state_dict(),strict=False)
    expected={n for n in generator.state_dict() if n.startswith(('spatial_projection.','enhancement.'))}
    if set(result.missing_keys)!=expected or result.unexpected_keys:raise ValueError('unaccounted initialization')
    for name,value in original.state_dict().items():
        if not torch.equal(value,generator.state_dict()[name]):raise ValueError('official migration tensor mismatch')
    if generator.spatial_projection.weight.count_nonzero():raise ValueError('spatial input must start zero')
    if memory=='full' and generator.enhancement.to_hidden.weight.count_nonzero():raise ValueError('cross branch must start zero')
    report.update(memory=memory,new_spatial_tensors=sorted(expected),fusion_initial_alpha=.5,
                  initialization='official_TiTok_plus_new_2d_spatial_fusion',local_pilot_resumed=False)
    return JointSystem(generator),report
