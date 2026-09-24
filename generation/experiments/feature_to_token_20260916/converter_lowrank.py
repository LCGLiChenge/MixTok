"""Low-rank spatial mapper for the frozen RGB-reencoded proxy vocabulary."""
import torch
from torch import nn


class LowRankFeatureTokenConverter(nn.Module):
    def __init__(self, channels=256, hidden=128, vocabulary=16384, rank=24):
        super().__init__()
        if not 0 < rank <= channels:
            raise ValueError("rank must be in (0, channels]")
        self.config = dict(channels=channels, hidden=hidden, vocabulary=vocabulary, rank=rank)
        self.residual = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, channels, 1))
        self.project = nn.Conv2d(channels, rank, 1, bias=False)
        self.head = nn.Conv2d(rank, vocabulary, 1)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    @torch.no_grad()
    def initialize_nearest(self, book, temperature=.01):
        if book.shape != (self.config["vocabulary"], self.config["channels"]):
            raise ValueError("projected codeword dimensions differ")
        # Best Frobenius-norm rank-r approximation to the original linear
        # nearest-codeword logits. Keep the exact squared-norm bias.
        scale = self.config["channels"] * temperature
        weight = (2 * book / scale).float()
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        r = self.config["rank"]
        singular_root = s[:r].sqrt()
        self.project.weight.copy_((singular_root[:, None] * vh[:r])[:, :, None, None])
        self.head.weight.copy_((u[:, :r] * singular_root[None, :])[:, :, None, None])
        self.head.bias.copy_((-book.float().square().sum(1) / scale))

    def forward(self, features):
        return self.head(self.project(features + self.residual(features)))

    @torch.no_grad()
    def tokens(self, features):
        return self(features).flatten(2).argmax(1)


def test():
    torch.manual_seed(23)
    model = LowRankFeatureTokenConverter(channels=8, hidden=8, vocabulary=31, rank=4)
    model.initialize_nearest(torch.randn(31, 8))
    x = torch.randn(2, 8, 4, 4)
    assert model(x).shape == (2, 31, 4, 4)
    loss = nn.functional.cross_entropy(model(x), torch.randint(31, (2, 4, 4)))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print("PASS low-rank spatial shape, initialization, finite backward")


if __name__ == "__main__":
    test()
