"""Frozen, asset-relative MoT/native TiTok/E117 with strict tensor provenance."""
from pathlib import Path
import sys
import torch
from torch import nn
from h20.assets import ROOT,specifications,digest
from h20_joint.routing import routed_grid


def load_vendor():
    for directory in ('TiTok','LlamaGen','E117'):
        path=str(ROOT/'third_party'/directory)
        if path not in sys.path:sys.path.insert(0,path)


class FrozenAssets:
    def __init__(self,root,device='cuda',chunk=4,keep_encoder=False):
        root=Path(root);self.device=torch.device(device);self.chunk=int(chunk)
        if self.chunk<1:raise ValueError('positive frozen chunk required')
        needed=('weights/mot_latest.pt','weights/tokenizer_titok_l32.bin','router/e117.pt')
        self.identities={}
        for spec in specifications():
            if spec['local_path'] in needed:
                path=root/spec['local_path']
                if path.stat().st_size!=spec['size'] or digest(path)!=spec['sha256']:
                    raise ValueError('frozen asset identity mismatch: '+spec['local_path'])
                self.identities[spec['local_path']]=spec['sha256']
        if len(self.identities)!=3:raise ValueError('incomplete frozen asset specification')
        load_vendor()
        from omegaconf import OmegaConf
        from modeling.titok import TiTok
        from vq_model import VQ_16
        from e117_ar_adapter import load_e117,E117ARDecisionAdapter
        from h20_joint.mot_latent import TiTokToLlamaGenLatentDecoder
        config=OmegaConf.load(ROOT/'third_party/TiTok/configs/infer/TiTok/titok_l32.yaml')
        native=TiTok(config)
        state=torch.load(root/'weights/tokenizer_titok_l32.bin',map_location='cpu',weights_only=True,mmap=True)
        if 'model' in state:state=state['model']
        native.load_state_dict(state,strict=True)
        for name,value in native.state_dict().items():
            if not torch.equal(value,state[name]):raise ValueError('native tokenizer tensor mismatch')
        native_count=len(state);del state
        trunk=TiTokToLlamaGenLatentDecoder(native.decoder)
        # MoT EMA contains no training-only codebook-usage history. Disable its
        # registration, keeping strict loading of EVERY learned VQ tensor.
        vq=VQ_16(codebook_size=16384,codebook_embed_dim=8,codebook_show_usage=False)
        mot=torch.load(root/'weights/mot_latest.pt',map_location='cpu',weights_only=True,mmap=True)
        if int(mot['step'])!=199440:raise ValueError('requires MoT199440 EMA')
        state=mot['model_ema'];retained=0
        for prefix,module in [('latent_decoder.',trunk),('llamagen_vq.',vq)]:
            subset={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
            module.load_state_dict(subset,strict=True)
            for name,value in module.state_dict().items():
                if not torch.equal(value,subset[name]):raise ValueError('MoT EMA tensor mismatch: '+prefix+name)
            retained+=len(subset)
        del state,mot,subset
        # These encoders are not used: training consumes packed codes, while
        # fusion's teacher is decoded from the SAME generated1D prefix.
        native.encoder=nn.Identity();native.latent_tokens=None
        if not keep_encoder:
            vq.encoder=nn.Identity();vq.quant_conv=nn.Identity()
        self.native=native.to(device).eval().requires_grad_(False)
        shell=nn.Module();shell.titok=nn.Module();shell.titok.quantize=self.native.quantize
        shell.latent_decoder=trunk;shell.llamagen_vq=vq
        self.shell=shell.to(device).eval().requires_grad_(False)
        student,metadata=load_e117(root/'router/e117.pt',torch.device('cpu'),use_ema=True)
        if metadata['loaded_model_state']!='model_ema':raise ValueError('E117 EMA required')
        for name,value in student.state_dict().items():
            if not torch.equal(value,metadata['model_ema'][name]):raise ValueError('E117 EMA tensor mismatch')
        self.audit=dict(native_tensors_exact=native_count,mot_tensors_exact=retained,router_tensors=len(student.state_dict()),
                        identities=self.identities,original_llamagen_checkpoint_used=False,codebook_show_usage=False)
        del metadata
        self.adapter=E117ARDecisionAdapter(self.shell,student).to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def features(self,ids):
        parts=[]
        with torch.autocast(ids.device.type,enabled=False):
            for codes in ids.split(self.chunk):
                q=self.adapter._quantized_from_codes(codes)
                parts.append(self.shell.latent_decoder(q))
        with torch.inference_mode(False):return torch.cat(parts).detach().clone()

    @torch.no_grad()
    def bundle(self,ids):
        parts={k:[] for k in ('base','scores','index','valid')}
        with torch.autocast(ids.device.type,enabled=False):
            for codes in ids.split(self.chunk):
                row=self.adapter(codes,return_1d_features=True)
                if any(row[k] for k in ('source_image_used','f_2d_used','probe_reconstructions','old_router_forwards')):
                    raise ValueError('Router used unavailable generation information')
                index=row['selected_indices_padded'];valid=row['selected_indices_valid']
                index=index.masked_fill(~valid,256).sort(1).values.masked_fill(~valid,-1)
                for key,value in dict(base=row['f_1d'],scores=row['grid_scores'],index=index,valid=valid).items():
                    parts[key].append(value)
        with torch.inference_mode(False):return {k:torch.cat(v).detach().clone() for k,v in parts.items()}

    @torch.no_grad()
    def mixed(self,base,codes,index,valid):
        routed_grid(index,valid)
        if ((codes[valid]<0)|(codes[valid]>=16384)).any():raise ValueError('invalid generated2D code')
        projection=self.shell.llamagen_vq.post_quant_conv
        if projection.kernel_size!=(1,1) or projection.stride!=(1,1) or projection.padding!=(0,0):
            raise ValueError('sparse projection must remain pointwise')
        emb=self.shell.llamagen_vq.quantize.get_codebook_entry(codes[valid])
        projected=projection(emb.t()[None,:,None,:]).squeeze(0).squeeze(1).t().contiguous()
        result=base.flatten(2).transpose(1,2).clone()
        batch=torch.arange(len(base),device=base.device)[:,None].expand_as(index)[valid]
        result[batch,index[valid]]=projected.to(result.dtype)
        return result.transpose(1,2).reshape_as(base)

    @torch.no_grad()
    def teacher(self,ids):
        with torch.autocast(ids.device.type,enabled=False):
            return torch.cat([self.native.decode_tokens(v).clamp(0,1) for v in ids.split(self.chunk)])

    def render(self,features):
        # Frozen weights; KEEP the graph to the trainable fusion features.
        with torch.autocast(features.device.type,enabled=False):
            return ((self.shell.llamagen_vq.decoder(features.float())+1)*.5).clamp(0,1)

    def assert_frozen(self):
        for module in (self.native,self.shell,self.adapter):
            if any(m.training for m in module.modules()) or any(p.requires_grad or p.grad is not None for p in module.parameters()):
                raise ValueError('frozen tokenizer/decoder/Router changed training ownership')
