"""Wrapper for profiling the sparse MoE MLP block.

Analogous to vidur.profiling.mlp.mlp_wrapper but for MoE layers.
Profiles: routing, grouped expert GEMM, shared expert, dispatch/combine.
"""

import os

import torch

from vidur.profiling.common.cuda_timer import CudaTimer

# Monkey-patch before importing sarathi
import sarathi.metrics.cuda_timer
sarathi.metrics.cuda_timer.CudaTimer = CudaTimer

from sarathi.model_executor.weight_utils import initialize_dummy_weights

from vidur.profiling.common.timer_stats_store import TimerStatsStore
from vidur.profiling.sparse.sparse_mlp_impl import SparseMoEModel
from vidur.profiling.utils import ProfileMethod
from vidur.profiling.utils.record_function_tracer import RecordFunctionTracer

WARMUP_STEPS = 2
ACTIVE_STEPS = 10


class SparseMlpWrapper:
    def __init__(
        self,
        hidden_size: int,
        num_routed_experts: int,
        num_experts_per_tok: int,
        expert_intermediate_size: int,
        num_shared_experts: int,
        shared_expert_intermediate_size: int,
        profile_method: str,
        output_dir: str,
    ):
        super().__init__()

        self.timer_stats_store = TimerStatsStore(profile_method=profile_method)

        self.hidden_size = hidden_size
        self.num_routed_experts = num_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.expert_intermediate_size = expert_intermediate_size
        self.num_shared_experts = num_shared_experts
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.profile_method = profile_method
        self.output_dir = output_dir
        os.makedirs(f"{self.output_dir}/profiler_traces/", exist_ok=True)

        self.model = SparseMoEModel(
            hidden_size=hidden_size,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            expert_intermediate_size=expert_intermediate_size,
            num_shared_experts=num_shared_experts,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            num_repeat_steps=(
                ACTIVE_STEPS
                if self.profile_method == ProfileMethod.RECORD_FUNCTION.value
                else 1
            ),
        )
        initialize_dummy_weights(self.model)
        self.model = self.model.to(dtype=torch.float16).cuda().eval()

    @torch.inference_mode()
    def profile(self, num_tokens: int):
        hidden_states = torch.randn(
            num_tokens, self.hidden_size,
            device="cuda", dtype=torch.float16,
        )

        if self.profile_method == ProfileMethod.RECORD_FUNCTION.value:
            # Warmup run (includes Triton autotune etc.)
            self.model(hidden_states)
            torch.cuda.synchronize()

            self.timer_stats_store.clear_stats()

            record_function_tracer = RecordFunctionTracer(self.output_dir)
            with record_function_tracer:
                self.model(hidden_states)

            time_stats = record_function_tracer.get_operation_time_stats()
        else:
            for _ in range(WARMUP_STEPS):
                self.model(hidden_states)

            torch.cuda.synchronize()
            self.timer_stats_store.clear_stats()

            for _ in range(ACTIVE_STEPS):
                self.model(hidden_states)

            torch.cuda.synchronize()
            time_stats = self.timer_stats_store.get_stats()

        stats = {
            "time_stats": time_stats,
            "hidden_size": self.hidden_size,
            "num_routed_experts": self.num_routed_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "expert_intermediate_size": self.expert_intermediate_size,
            "num_shared_experts": self.num_shared_experts,
            "num_tokens": num_tokens,
        }
        self.timer_stats_store.clear_stats()

        return stats
