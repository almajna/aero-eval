# Task: Implement `GroupedQueryAttention.forward`

You are implementing the body of the `forward` method on the
`GroupedQueryAttention` `nn.Module` defined below. Follow the contract
exactly.

This is **grouped-query attention** (GQA), used by Llama 2/3, Mistral, and
similar modern decoder LLMs. Queries have `n_heads` independent heads, but
keys and values share across groups: there are only `n_kv_heads`, with
each KV head consumed by `n_heads // n_kv_heads` consecutive query heads.

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GroupedQueryAttention(nn.Module):
    """Grouped-query attention.

    Constructor sets:
        - d_model:     total model dimension
        - n_heads:     number of query heads
        - n_kv_heads:  number of key/value heads (must divide n_heads)
        - head_dim:    d_model // n_heads
        - q_proj:  nn.Linear(d_model, n_heads * head_dim)        == Linear(d_model, d_model)
        - k_proj:  nn.Linear(d_model, n_kv_heads * head_dim)     != Linear(d_model, d_model)
        - v_proj:  nn.Linear(d_model, n_kv_heads * head_dim)     != Linear(d_model, d_model)
        - out_proj: nn.Linear(d_model, d_model)

    All projections are pre-defined for you below. Do not change __init__.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
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
      before softmax). The mask broadcasts across heads.
- **Output:** shape `[batch, seq, d_model]`, dtype float32.
- **Behavior:** standard scaled-dot-product attention with softmax
  temperature `1 / sqrt(head_dim)`, applied along the key axis. Each query
  head `i` attends to KV head `i // n_rep`. Concretely: project Q to
  `[bsz, seq, n_heads, head_dim]`, project K and V each to
  `[bsz, seq, n_kv_heads, head_dim]`, expand K and V along the head axis
  by `n_rep` so they align with the query head count, then run attention.
- **No new parameters.** Use only the four `nn.Linear` projections defined
  in `__init__`.

## Output format — strict

Return **only** the body of `forward`, as raw Python code, no markdown
fences, no surrounding prose, no class header. The first line of your
response must be valid Python at one level of indentation (matching the
indentation of `def forward`'s body inside the class). Example shape:

```
        bsz, seq, _ = q.shape
        q_h = self.q_proj(q).view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)
        ...
        return self.out_proj(out)
```

Do not include `def forward(...)`, do not include comments explaining the
math, do not include test code. Body only.
