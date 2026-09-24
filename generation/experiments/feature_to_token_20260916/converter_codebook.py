"""Map frozen 1D features into the native 8D LlamaGen codebook space."""
import torch
import torch.nn.functional as F
from torch import nn


class CodebookFeatureTokenConverter(nn.Module):
    def __init__(self, channels=256, hidden=128, vocabulary=16384,
                 code_dim=8, temperature=.01):
        super().__init__()
        self.config = dict(channels=channels, hidden=hidden, vocabulary=vocabulary,
                           code_dim=code_dim, temperature=temperature)
        self.residual = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, channels, 1))
        self.project = nn.Conv2d(channels, code_dim, 1)
        self.register_buffer("codebook", torch.empty(vocabulary, code_dim), persistent=False)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    @torch.no_grad()
    def initialize_codebook(self, native_book, post_quant_conv):
        expected = (self.config["vocabulary"], self.config["code_dim"])
        if native_book.shape != expected:
            raise ValueError(f"native codebook dimensions differ: {native_book.shape} != {expected}")
        weight = post_quant_conv.weight.detach().float().flatten(1)
        bias = post_quant_conv.bias.detach().float()
        if weight.shape != (self.config["channels"], self.config["code_dim"]):
            raise ValueError("post-quant projection dimensions differ")
        inverse = torch.linalg.pinv(weight)
        self.project.weight.copy_(inverse[:, :, None, None])
        self.project.bias.copy_(-(inverse @ bias))
        self.codebook.copy_(native_book.detach().float())

    def forward(self, features):
        code = self.project(features + self.residual(features))
        scale = self.config["code_dim"] * self.config["temperature"]
        weight = (2 * self.codebook / scale)[:, :, None, None]
        bias = -self.codebook.square().sum(1) / scale
        return F.conv2d(code, weight, bias)

    @torch.no_grad()
    def tokens(self, features):
        return self(features).flatten(2).argmax(1)


def test():
    torch.manual_seed(23)
    channels, code_dim, vocabulary = 12, 4, 31
    model = CodebookFeatureTokenConverter(channels=channels, hidden=8,
                                           vocabulary=vocabulary, code_dim=code_dim)
    projection = nn.Conv2d(code_dim, channels, 1)
    native = torch.randn(vocabulary, code_dim)
    model.initialize_codebook(native, projection)
    projected = projection(native.T.unsqueeze(0).unsqueeze(-1)).squeeze(0).squeeze(-1).T
    recovered = model.project(projected.T.unsqueeze(0).unsqueeze(-1)).squeeze(0).squeeze(-1).T
    torch.testing.assert_close(recovered, native, atol=2e-5, rtol=2e-5)
    x = torch.randn(2, channels, 4, 4)
    assert model(x).shape == (2, vocabulary, 4, 4)
    loss = F.cross_entropy(model(x), torch.randint(vocabulary, (2, 4, 4)))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print("PASS affine inverse, fixed codebook logits, finite backward")


if __name__ == "__main__":
    test()
