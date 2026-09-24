"""Small learned latent acceptance head; all original models remain frozen."""
import torch
from torch import nn
from h20_joint.routing import routed_grid, score_values

ARMS = ('scalar', 'zero', 'aligned', 'shuffled')


class AcceptanceGate(nn.Module):
    def __init__(self, arm, channels=256, hidden=32):
        super().__init__()
        if arm not in ARMS or channels < 1 or hidden < 1: raise ValueError('invalid gate architecture')
        self.arm = arm; self.channels = channels; self.hidden = hidden
        if arm == 'scalar':
            self.logit = nn.Parameter(torch.zeros(()))
        else:
            # base, generated delta, score, selected mask, fixed x/y coordinates.
            self.input = nn.Conv2d(channels*2+4, hidden, 1)
            self.spatial = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
            self.output = nn.Conv2d(hidden, 1, 1)
            nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
            self.activation = nn.GELU()

    def forward(self, base, mixed, scores, prefix, index, valid):
        if (base.shape != mixed.shape or base.shape != (len(prefix), self.channels, 16, 16)
                or not torch.isfinite(base).all() or not torch.isfinite(mixed).all()):
            raise ValueError('requires finite base/generated mixed features')
        selected = routed_grid(index, valid).reshape(-1, 1, 16, 16)
        if scores.shape != selected.shape or not torch.isfinite(scores).all(): raise ValueError('invalid score map')
        if prefix.shape != (len(base), 32) or prefix.dtype != torch.long or bool(((prefix < 0) | (prefix >= 4096)).any()):
            raise ValueError('requires completed1D codes')
        # Detached ordinary tensors; caller must convert inference tensors before
        # training because projection backward needs to save these constants.
        base = base.detach(); delta = mixed.detach()-base
        if bool(delta.masked_select(~selected.expand_as(delta)).any()):
            raise ValueError('mixed features changed an unselected cell')
        if self.arm == 'scalar':
            logits = self.logit.expand(len(base), 1, 16, 16)
        else:
            score = score_values(scores, prefix, self.arm).reshape(-1, 1, 16, 16)
            axis = torch.arange(16, dtype=base.dtype, device=base.device)/7.5-1
            yy, xx = torch.meshgrid(axis, axis, indexing='ij')
            position = torch.stack((xx, yy))[None].expand(len(base), -1, -1, -1)
            inputs = torch.cat((base, delta, score.to(base.dtype), selected.to(base.dtype), position), 1)
            hidden = self.activation(self.input(inputs))
            hidden = self.activation(hidden+self.spatial(hidden))
            logits = self.output(hidden)
        alpha = logits.sigmoid()*selected.to(logits.dtype)
        return base+alpha.to(base.dtype)*delta, alpha


def make_gates(seed=20261204, channels=256, hidden=32, device='cpu'):
    # Equal spatial architectures start with identical weights. Preserve caller RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = AcceptanceGate('zero', channels, hidden)
        gates = {'scalar': AcceptanceGate('scalar', channels, hidden)}
        for arm in ARMS[1:]:
            gate = AcceptanceGate(arm, channels, hidden)
            gate.load_state_dict(reference.state_dict(), strict=True); gates[arm] = gate
    return {name: gate.to(device) for name, gate in gates.items()}


def normal_conditions(*values):
    """No inference-tensor state leaks across the frozen/trainable boundary."""
    with torch.inference_mode(False), torch.no_grad():
        return tuple(value.detach().clone() for value in values)
