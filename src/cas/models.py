"""Feed-forward ReLU networks and the handcrafted hierarchical-equality network.

Architecture (Geiger et al. 2024 §4.1): h_0 = ReLU(x W_0 + b_0), h_l = ReLU(h_{l-1} W_l + b_l),
logits = h_{L-1} W_out + b_out.  "Layer l" always means the post-ReLU output h_l of hidden layer l
(pyvene's ``block_output`` of block l), with l = 0 the first hidden layer.  The class index 1 means
the task label True.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, d_in: int, width: int, n_layers: int, n_classes: int = 2):
        super().__init__()
        self.d_in, self.width, self.n_layers = d_in, width, n_layers
        self.layers = nn.ModuleList(
            [nn.Linear(d_in if i == 0 else width, width) for i in range(n_layers)])
        self.head = nn.Linear(width, n_classes)

    def hidden(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        """Post-ReLU activations of hidden layer ``layer``."""
        h = x
        for i in range(layer + 1):
            h = torch.relu(self.layers[i](h))
        return h

    def from_hidden(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Logits computed from (possibly intervened) activations of hidden layer ``layer``."""
        for i in range(layer + 1, self.n_layers):
            h = torch.relu(self.layers[i](h))
        return self.head(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.from_hidden(self.hidden(x, self.n_layers - 1), self.n_layers - 1)


def handcrafted_hierarchical_equality(emb_dim: int = 2) -> MLP:
    """The handcrafted network from the pyvene DAS tutorial (embedding_dim = 2, width 8).

    Layer 0 computes [ReLU(a-b), ReLU(b-a), ReLU(c-d), ReLU(d-c)] so neurons 0..3 encode
    |a-b| (variable WX) and neurons 4..7 encode |c-d| (variable YZ).  Layer 1 computes
    [ReLU(|a-b|-|c-d|), ReLU(|c-d|-|a-b|), |c-d|, |a-b|, 0, ...] and the head outputs
    [||a-b|-|c-d|| - 0.999999(|a-b|+|c-d|), 1e-14], so class 1 (True) wins iff both pairs are
    equal or both are different.  Weights are copied from the tutorial (pyvene's Linear stores
    W as (out, in); W2 and W3 were given as (in, out) and transposed there).
    """
    assert emb_dim == 2, "the tutorial's weights are for 2-dimensional objects"
    w1 = [[1, 0, -1, 0, 0, 0, 0, 0], [0, 1, 0, -1, 0, 0, 0, 0],
          [-1, 0, 1, 0, 0, 0, 0, 0], [0, -1, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, -1, 0], [0, 0, 0, 0, 0, 1, 0, -1],
          [0, 0, 0, 0, -1, 0, 1, 0], [0, 0, 0, 0, 0, -1, 0, 1]]
    w2 = [[1, -1, 0, 1, 0, 0, 0, 0]] * 4 + [[-1, 1, 1, 0, 0, 0, 0, 0]] * 4
    w3 = [[1, 0], [1, 0], [-0.999999, 0], [-0.999999, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    m = MLP(d_in=8, width=8, n_layers=2)
    with torch.no_grad():
        m.layers[0].weight.copy_(torch.tensor(w1, dtype=torch.float32))
        m.layers[0].bias.zero_()
        m.layers[1].weight.copy_(torch.tensor(w2, dtype=torch.float32).T)
        m.layers[1].bias.zero_()
        m.head.weight.copy_(torch.tensor(w3, dtype=torch.float32).T)
        m.head.bias.copy_(torch.tensor([0.0, 1e-14]))
    return m


def save_network(model: MLP, meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta, d_in=model.d_in, width=model.width, n_layers=model.n_layers)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path.with_suffix(".pt"))
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load_network(path: Path, device: str = "cpu") -> tuple[MLP, dict]:
    blob = torch.load(path.with_suffix(".pt"), map_location=device, weights_only=False)
    meta = blob["meta"]
    model = MLP(meta["d_in"], meta["width"], meta["n_layers"]).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, meta
