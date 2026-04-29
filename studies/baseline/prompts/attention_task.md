# Task: Implement `CustomAttention.forward`

You are implementing the body of the `forward` method on the
`CustomAttention` `nn.Module` defined below. Follow the contract exactly.

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CustomAttention(nn.Module):
    """Multi-head scaled dot-product attention.

    Constructor sets:
        - d_model:  total model dimension (must be divisible by n_heads)
        - n_heads:  number of attention heads
        - head_dim: d_model // n_heads
        - q_proj, k_proj, v_proj, out_proj: nn.Linear(d_model, d_model)

    All projections are pre-defined for you below. Do not change __init__.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        # IMPLEMENT THIS METHOD.
        ...
```

## Forward contract — strict

- **Inputs:**
    - `q`, `k`, `v`: shape `[batch, seq, d_model]`, dtype float32.
    - `mask`: shape `[batch, seq, seq]` of booleans **or** `None`. If
      provided, `True` = attend, `False` = mask out (set to large negative
      before softmax).
- **Output:** shape `[batch, seq, d_model]`, dtype float32.
- **Behavior:** standard multi-head scaled dot-product attention, with
  softmax temperature `1 / sqrt(head_dim)`, applied along the key axis.
- **No new parameters.** Use only the four `nn.Linear` projections defined
  in `__init__`.

## Output format — strict

Return **only** the body of `forward`, as a Python code block, no markdown
fences, no surrounding prose, no class header. The first line of your
response must be valid Python at one level of indentation
(matching `def forward`). Example shape:

```
        bsz, seq, _ = q.shape
        q_proj = self.q_proj(q).view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)
        ...
        return self.out_proj(out)
```

Do not include `def forward(...)`, do not include comments explaining the
math, do not include test code. Body only.
