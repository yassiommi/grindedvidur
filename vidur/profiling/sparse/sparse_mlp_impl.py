"""Sparse (MoE) MLP profiling model.

Builds a single MoE transformer block that mirrors DeepSeek-V3 / Mixtral
architecture for on-device profiling:

  1. Router/gating network   (hidden -> num_experts linear)
  2. Top-K expert selection   (torch.topk)
  3. Routed expert GEMMs      (grouped GEMM via loop over active experts)
  4. Shared expert MLP        (dense up/down with gated activation)
  5. Expert combine            (scatter-add results back)

Each operation is wrapped in CudaTimer for profiling.
"""

from typing import Optional

import torch

from vidur.profiling.common.cuda_timer import CudaTimer


class MoERouter(torch.nn.Module):
    """Gating network: projects hidden states to expert logits + top-K selection."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.gate = torch.nn.Linear(hidden_size, num_experts, bias=False)
        self.top_k = top_k

        self._router_gate_timer = CudaTimer("moe_router_gate")
        self._router_topk_timer = CudaTimer("moe_router_topk")
        self._router_softmax_timer = CudaTimer("moe_router_softmax")

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: [num_tokens, hidden_size]
        with self._router_gate_timer:
            logits = self.gate(hidden_states)  # [num_tokens, num_experts]
        with self._router_softmax_timer:
            weights = torch.softmax(logits, dim=-1)
        with self._router_topk_timer:
            topk_weights, topk_indices = torch.topk(weights, self.top_k, dim=-1)
            # Normalize top-K weights
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights, topk_indices


class ExpertMLP(torch.nn.Module):
    """Single expert: gated MLP (gate_proj * up_proj -> silu -> down_proj)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class SharedExpertMLP(torch.nn.Module):
    """Shared expert: same structure as ExpertMLP but always activated."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class MoEBlock(torch.nn.Module):
    """Full MoE block: router + routed experts (grouped GEMM) + shared experts.

    The routed expert computation simulates grouped GEMM by running each
    active expert on its assigned token subset.  This captures the real
    kernel launch and memory access patterns of sparse MoE inference.
    """

    def __init__(
        self,
        hidden_size: int,
        num_routed_experts: int,
        num_experts_per_tok: int,
        expert_intermediate_size: int,
        num_shared_experts: int = 0,
        shared_expert_intermediate_size: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_routed_experts = num_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts

        # Router
        self.router = MoERouter(hidden_size, num_routed_experts, num_experts_per_tok)

        # Routed experts — we only instantiate num_experts_per_tok experts
        # since that's the maximum that execute per token.  For profiling
        # what matters is the GEMM shapes, not having all 256 expert weights.
        self.routed_experts = torch.nn.ModuleList([
            ExpertMLP(hidden_size, expert_intermediate_size)
            for _ in range(min(num_routed_experts, 64))  # cap at 64 for memory
        ])

        # Shared experts
        if num_shared_experts > 0:
            se_size = shared_expert_intermediate_size or expert_intermediate_size
            self.shared_experts = torch.nn.ModuleList([
                SharedExpertMLP(hidden_size, se_size)
                for _ in range(num_shared_experts)
            ])
        else:
            self.shared_experts = torch.nn.ModuleList()

        # Timers
        self._expert_dispatch_timer = CudaTimer("moe_expert_dispatch")
        self._expert_gemm_timer = CudaTimer("moe_expert_gemm")
        self._expert_combine_timer = CudaTimer("moe_expert_combine")
        self._shared_expert_timer = CudaTimer("moe_shared_expert")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        residual = hidden_states

        # 1. Route
        topk_weights, topk_indices = self.router(hidden_states)
        # topk_weights: [num_tokens, top_k], topk_indices: [num_tokens, top_k]

        # 2. Dispatch: organize tokens by expert assignment
        with self._expert_dispatch_timer:
            # Flatten to get per-token-expert pairs
            flat_indices = topk_indices.view(-1)  # [num_tokens * top_k]
            flat_weights = topk_weights.view(-1)  # [num_tokens * top_k]
            # Expand hidden states for each expert assignment
            expanded_hidden = hidden_states.unsqueeze(1).expand(
                -1, self.num_experts_per_tok, -1
            ).reshape(-1, self.hidden_size)  # [num_tokens * top_k, hidden]

        # 3. Grouped GEMM: run each active expert on its assigned tokens
        with self._expert_gemm_timer:
            expert_outputs = torch.zeros_like(expanded_hidden)
            num_actual_experts = len(self.routed_experts)
            for expert_idx in range(num_actual_experts):
                mask = (flat_indices == expert_idx)
                if mask.any():
                    expert_input = expanded_hidden[mask]
                    expert_outputs[mask] = self.routed_experts[expert_idx](expert_input)
            # Handle experts beyond our instantiated set by reusing expert 0
            if self.num_routed_experts > num_actual_experts:
                remaining_mask = (flat_indices >= num_actual_experts)
                if remaining_mask.any():
                    expert_input = expanded_hidden[remaining_mask]
                    expert_outputs[remaining_mask] = self.routed_experts[0](expert_input)

        # 4. Combine: weighted sum of expert outputs back to token space
        with self._expert_combine_timer:
            weighted_outputs = expert_outputs * flat_weights.unsqueeze(-1)
            combined = weighted_outputs.view(
                num_tokens, self.num_experts_per_tok, self.hidden_size
            ).sum(dim=1)  # [num_tokens, hidden]

        # 5. Shared experts (always activated, run on all tokens)
        with self._shared_expert_timer:
            shared_output = torch.zeros_like(hidden_states)
            for shared_expert in self.shared_experts:
                shared_output = shared_output + shared_expert(hidden_states)

        return combined + shared_output


class SparseMoEModel(torch.nn.Module):
    """Full sparse transformer block for profiling: norm -> MoE -> residual.

    This wraps the MoE block with the surrounding layer norm and residual
    connection, matching the actual DeepSeek-V3 transformer block structure.
    """

    def __init__(
        self,
        hidden_size: int,
        num_routed_experts: int,
        num_experts_per_tok: int,
        expert_intermediate_size: int,
        num_shared_experts: int = 0,
        shared_expert_intermediate_size: Optional[int] = None,
        num_repeat_steps: int = 1,
    ):
        super().__init__()
        self.num_repeat_steps = num_repeat_steps

        self.norm = torch.nn.LayerNorm(hidden_size)
        self.moe = MoEBlock(
            hidden_size=hidden_size,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            expert_intermediate_size=expert_intermediate_size,
            num_shared_experts=num_shared_experts,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
        )

        self._norm_timer = CudaTimer("moe_block_norm")
        self._residual_timer = CudaTimer("moe_block_residual")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_repeat_steps):
            residual = hidden_states
            with self._norm_timer:
                hidden_states = self.norm(hidden_states)
            hidden_states = self.moe(hidden_states)
            with self._residual_timer:
                hidden_states = hidden_states + residual
        return hidden_states
