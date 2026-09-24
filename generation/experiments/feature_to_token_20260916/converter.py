"""Small spatial classifier initialized to native projected-codeword distance."""
import torch
from torch import nn


class FeatureTokenConverter(nn.Module):
    def __init__(self, channels=256, hidden=128, vocabulary=16384):
        super().__init__()
        self.config = dict(channels=channels, hidden=hidden, vocabulary=vocabulary)
        self.residual = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, channels, 1))
        self.head = nn.Conv2d(channels, vocabulary, 1)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    @torch.no_grad()
    def initialize_nearest(self, book, temperature=.01):
        if book.shape != (self.config['vocabulary'], self.config['channels']):
            raise ValueError('projected codeword dimensions differ')
        scale = self.config['channels'] * temperature
        self.head.weight.copy_((2 * book / scale)[:, :, None, None])
        self.head.bias.copy_(-book.square().sum(1) / scale)

    def forward(self, features):
        return self.head(features + self.residual(features))

    @torch.no_grad()
    def tokens(self, features):
        return self(features).flatten(2).argmax(1)


def load_converter(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['format'] != 'feature_token_converter_v1':
        raise ValueError('wrong converter checkpoint')
    model = FeatureTokenConverter(**state['model_config'])
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval().requires_grad_(False), state


def test():
    from experiments.feature_to_token_20260916.probe import nearest
    torch.manual_seed(23)
    model = FeatureTokenConverter(channels=8, hidden=8, vocabulary=31)
    book, x = torch.randn(31, 8), torch.randn(2, 8, 4, 4)
    model.initialize_nearest(book)
    ids, _ = nearest(x, book)
    assert torch.equal(model.tokens(x), ids)
    target = torch.randint(31, (2, 4, 4))
    loss = nn.functional.cross_entropy(model(x), target)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print('PASS nearest initialization, spatial shape, finite backward')


if __name__ == '__main__': test()
