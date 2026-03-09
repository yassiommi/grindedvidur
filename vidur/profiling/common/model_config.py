from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
from sarathi.config import ParallelConfig

from vidur.config.model_config import BaseModelConfig
from vidur.types import ActivationType, NormType


class ModelConfig:
    def __init__(
        self,
        name: str,
        num_layers: int,
        num_q_heads: int,
        num_kv_heads: int,
        embedding_dim: int,
        mlp_hidden_dim: int,
        max_position_embeddings: int,
        use_gated_mlp: bool,
        use_bias: bool,
        use_qkv_bias: bool,
        activation: ActivationType,
        norm: NormType,
        post_attn_norm: bool,
        vocab_size: int,
        is_neox_style: Optional[bool] = True,
        rope_theta: Optional[int] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        partial_rotary_factor: float = 1.0,
        no_tensor_parallel: bool = False,
        # MoE fields
        is_moe: bool = False,
        num_routed_experts: int = 1,
        num_experts_per_tok: int = 1,
        num_shared_experts: int = 0,
        moe_intermediate_size: Optional[int] = None,
        # MLA fields
        attention_type: str = "MHA",
        kv_lora_rank: Optional[int] = None,
        q_lora_rank: Optional[int] = None,
        qk_nope_head_dim: Optional[int] = None,
        qk_rope_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        # Engram fields (passed through, not used by profiler directly)
        **kwargs,
    ):
        self.name = name
        self.num_layers = num_layers
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.embedding_dim = embedding_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.max_position_embeddings = max_position_embeddings
        self.use_gated_mlp = use_gated_mlp
        self.vocab_size = vocab_size
        self.use_bias = use_bias
        self.use_qkv_bias = use_qkv_bias
        self.activation = str(activation)
        self.norm = str(norm)
        self.post_attn_norm = post_attn_norm
        self.no_tensor_parallel = no_tensor_parallel
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.is_neox_style = is_neox_style

        # MoE fields
        self.is_moe = is_moe
        self.num_routed_experts = num_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.moe_intermediate_size = moe_intermediate_size

        # MLA fields
        self.attention_type = attention_type
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim

        assert self.norm in ["layer_norm", "rms_norm"]
        assert self.activation in ["gelu", "silu"]

        if self.use_gated_mlp:
            assert self.activation == "silu"
        else:
            assert self.activation == "gelu"

    @property
    def expert_intermediate_size(self) -> int:
        if self.moe_intermediate_size is not None:
            return self.moe_intermediate_size
        return self.mlp_hidden_dim

    @property
    def has_mla(self) -> bool:
        return self.attention_type == "MLA" and self.kv_lora_rank is not None

    @staticmethod
    def from_model_name(model_name: str):
        model_config: BaseModelConfig = BaseModelConfig.create_from_name(model_name)
        model_config_dict = asdict(model_config)

        return ModelConfig(model_name, **model_config_dict)

    def get_num_q_heads(self, parallel_config: ParallelConfig):
        return self.num_q_heads // parallel_config.tensor_parallel_size

    def get_num_kv_heads(self, parallel_config: ParallelConfig):
        return self.num_kv_heads // parallel_config.tensor_parallel_size

    def get_head_size(self):
        return self.embedding_dim // self.num_q_heads

    @property
    def dtype(self):
        return torch.float16
