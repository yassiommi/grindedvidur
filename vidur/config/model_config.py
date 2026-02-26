from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from vidur.config.base_fixed_config import BaseFixedConfig
from vidur.logger import init_logger
from vidur.types import ActivationType, NormType

logger = init_logger(__name__)


@dataclass
class BaseModelConfig(BaseFixedConfig):
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    embedding_dim: int
    mlp_hidden_dim: int
    max_position_embeddings: int
    use_gated_mlp: bool
    use_bias: bool
    use_qkv_bias: bool
    activation: ActivationType
    norm: NormType
    post_attn_norm: bool
    vocab_size: int
    is_neox_style: Optional[bool] = True
    rope_theta: Optional[float] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 1.0
    no_tensor_parallel: bool = False

    # MoE fields (defaults for dense models)
    is_moe: bool = False
    num_routed_experts: int = 1
    num_experts_per_tok: int = 1
    num_shared_experts: int = 0
    moe_intermediate_size: Optional[int] = None  # None means use mlp_hidden_dim

    # MLA (Multi-head Latent Attention) fields for DeepSeek
    attention_type: str = "MHA"  # "MHA", "GQA", or "MLA"
    kv_lora_rank: Optional[int] = None
    q_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None

    @property
    def expert_intermediate_size(self) -> int:
        """FFN hidden dim used for expert layers."""
        if self.moe_intermediate_size is not None:
            return self.moe_intermediate_size
        return self.mlp_hidden_dim

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.num_q_heads

    @classmethod
    def get_profiling_name(cls) -> str:
        """Model name used for profiling data file lookup.

        Dense models use their own name.  MoE models fall back to a
        similar dense model whose profiling data is available.
        Override in MoE subclasses.
        """
        return cls.get_name()

    @classmethod
    def get_profiling_config(cls) -> "BaseModelConfig":
        """Return the model config used for profiling data filtering.

        Dense models return their own config.  MoE models return the
        fallback model's config so that the profiling data filter
        (n_head, n_embd, etc.) matches the available profiling data.
        """
        profiling_name = cls.get_profiling_name()
        if profiling_name != cls.get_name():
            return BaseModelConfig.create_from_name(profiling_name)
        return cls()


@dataclass
class Llama2ModelConfig(BaseModelConfig):
    max_position_embeddings: int = 16384
    use_gated_mlp: bool = True
    use_bias: bool = False
    use_qkv_bias: bool = False
    activation: ActivationType = ActivationType.SILU
    norm: NormType = NormType.RMS_NORM
    post_attn_norm: bool = True
    vocab_size: int = 32768
    is_neox_style: Optional[bool] = True
    rope_theta: Optional[float] = 10000
    rope_scaling: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 1.0
    no_tensor_parallel: bool = False

    @staticmethod
    def get_name():
        return "meta-llama/Llama-2-Config"


@dataclass
class CodeLlama34BModelConfig(Llama2ModelConfig):
    num_layers: int = 48
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 22016
    rope_theta: Optional[float] = 1000000

    @staticmethod
    def get_name():
        return "codellama/CodeLlama-34b-Instruct-hf"


@dataclass
class Llama2_7BModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 32
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 11008
    max_position_embeddings: int = 4096

    @staticmethod
    def get_name():
        return "meta-llama/Llama-2-7b-hf"


@dataclass
class Llama2_70BModelConfig(Llama2ModelConfig):
    num_layers: int = 80
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 28672
    max_position_embeddings: int = 4096

    @staticmethod
    def get_name():
        return "meta-llama/Llama-2-70b-hf"


@dataclass
class Llama3_8BModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 8
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 14336
    max_position_embeddings: int = 4096
    rope_theta: Optional[float] = 500000
    vocab_size: int = 128256

    @staticmethod
    def get_name():
        return "meta-llama/Meta-Llama-3-8B"


@dataclass
class Llama3_70BModelConfig(Llama2ModelConfig):
    num_layers: int = 80
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 28672
    max_position_embeddings: int = 8192
    rope_theta: Optional[float] = 500000
    vocab_size: int = 128256

    @staticmethod
    def get_name():
        return "meta-llama/Meta-Llama-3-70B"


@dataclass
class InternLMModelConfig(Llama2ModelConfig):
    max_position_embeddings: int = 4096
    vocab_size: int = 103168


@dataclass
class InternLM_20BModelConfig(InternLMModelConfig):
    num_layers: int = 60
    num_q_heads: int = 40
    num_kv_heads: int = 40
    embedding_dim: int = 5120
    mlp_hidden_dim: int = 13824

    @staticmethod
    def get_name():
        return "internlm/internlm-20b"


@dataclass
class InternLM2ModelConfig(Llama2ModelConfig):
    max_position_embeddings: int = 32768
    vocab_size: int = 92544


@dataclass
class InternLM2_20BModelConfig(InternLM2ModelConfig):
    num_layers: int = 48
    num_q_heads: int = 48
    num_kv_heads: int = 8
    embedding_dim: int = 6144
    mlp_hidden_dim: int = 16384
    rope_theta: Optional[float] = 1000000

    @staticmethod
    def get_name():
        return "internlm/internlm2-20b"


@dataclass
class Phi2ModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 32
    embedding_dim: int = 2560
    mlp_hidden_dim: int = 10240
    max_position_embeddings: int = 2048
    use_gated_mlp: bool = False
    use_bias: bool = True
    use_qkv_bias: bool = True
    activation: ActivationType = ActivationType.GELU
    norm: NormType = NormType.LAYER_NORM
    post_attn_norm: bool = False
    vocab_size: int = 51200
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: Optional[float] = 10000
    partial_rotary_factor: float = 0.4
    no_tensor_parallel: bool = True

    @staticmethod
    def get_name():
        return "microsoft/phi-2"


@dataclass
class QwenModelConfig(Llama2ModelConfig):
    use_qkv_bias: bool = True
    max_position_embeddings: int = 32768
    vocab_size: int = 152064

    @staticmethod
    def get_name():
        return "Qwen/Qwen-Config"


@dataclass
class Qwen72BModelConfig(QwenModelConfig):
    num_layers: int = 80
    num_q_heads: int = 64
    num_kv_heads: int = 64
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 24576
    rope_theta: Optional[float] = 1000000

    @staticmethod
    def get_name():
        return "Qwen/Qwen-72B"


# ============================================================
# MoE Model Configs (based on InferSim model specifications)
# ============================================================


@dataclass
class DeepSeekV3ModelConfig(BaseModelConfig):
    """DeepSeek-V3 with MLA attention and 256-expert MoE.

    Architecture uses Multi-head Latent Attention (MLA) with KV LoRA
    compression and sparse Mixture-of-Experts with 256 routed experts
    plus 1 shared expert per layer.
    """
    num_layers: int = 61
    num_q_heads: int = 128
    num_kv_heads: int = 128
    embedding_dim: int = 7168
    mlp_hidden_dim: int = 18432  # dense layers (first layer)
    max_position_embeddings: int = 131072
    use_gated_mlp: bool = True
    use_bias: bool = False
    use_qkv_bias: bool = False
    activation: ActivationType = ActivationType.SILU
    norm: NormType = NormType.RMS_NORM
    post_attn_norm: bool = True
    vocab_size: int = 129280
    rope_theta: Optional[float] = 10000

    # MoE
    is_moe: bool = True
    num_routed_experts: int = 256
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    moe_intermediate_size: int = 2048

    # MLA (Multi-head Latent Attention)
    attention_type: str = "MLA"
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    @staticmethod
    def get_name():
        return "deepseek-ai/DeepSeek-V3"

    @classmethod
    def get_profiling_name(cls) -> str:
        return "meta-llama/Meta-Llama-3-70B"


# ============================================================
# Engram Model Configs (Conditional Memory via Scalable Lookup)
#
# From "Conditional Memory via Scalable Lookup: A New Axis of
# Sparsity for Large Language Models" (arXiv:2601.07372).
# Engram adds a deterministic N-gram memory module to MoE,
# reallocating sparse parameters from routed experts to a
# static embedding table with O(1) lookup.
# ============================================================


@dataclass
class EngramBaseModelConfig(BaseModelConfig):
    """Base config for Engram paper models (30-block backbone, hidden=2560).

    All Engram models share a 30-block Transformer backbone with:
    - Hidden size 2560, 32 attention heads
    - Multi-head Latent Attention (MLA) via mHC (expansion rate 4)
    - DeepSeek-V3 tokenizer (128k vocab)
    """
    num_layers: int = 30
    num_q_heads: int = 32
    num_kv_heads: int = 32
    embedding_dim: int = 2560
    mlp_hidden_dim: int = 10240  # expansion rate 4
    max_position_embeddings: int = 32768
    use_gated_mlp: bool = True
    use_bias: bool = False
    use_qkv_bias: bool = False
    activation: ActivationType = ActivationType.SILU
    norm: NormType = NormType.RMS_NORM
    post_attn_norm: bool = True
    vocab_size: int = 129280  # DeepSeek-V3 tokenizer
    rope_theta: Optional[float] = 10000

    # Engram-specific fields
    has_engram: bool = False
    engram_layers: List = field(default_factory=list)  # layer indices with Engram module
    engram_num_heads: int = 0          # number of hash heads (H)
    engram_dim: int = 0                # embedding dimension per head
    engram_ngram_sizes: List = field(default_factory=list)  # N-gram orders [2,3]
    engram_total_params_b: float = 0.0  # total Engram table size in billions
    engram_compressed_vocab_size: int = 0  # vocabulary after tokenizer compression

    @staticmethod
    def get_name():
        return "deepseek-ai/Engram-Base"

    @classmethod
    def get_profiling_name(cls) -> str:
        return "microsoft/phi-2"


@dataclass
class Engram27BModelConfig(EngramBaseModelConfig):
    """Engram-27B: MoE + Conditional Memory (26.7B total params).

    Derived from MoE-27B by reducing routed experts from 72 to 55
    and reallocating freed parameters to a 5.7B Engram memory module
    (rho=74.3%). Engram module inserted at layers 2 and 15 with
    N={2,3}, 8 heads, dimension 1280.
    """
    # MoE (reduced from 72 to 55 routed experts)
    is_moe: bool = True
    num_routed_experts: int = 55
    num_experts_per_tok: int = 6
    num_shared_experts: int = 2
    moe_intermediate_size: int = 2560  # expert FFN hidden dim

    # Engram module
    has_engram: bool = True
    engram_layers: List = field(default_factory=lambda: [2, 15])
    engram_num_heads: int = 8
    engram_dim: int = 1280
    engram_ngram_sizes: List = field(default_factory=lambda: [2, 3])
    engram_total_params_b: float = 5.7
    engram_compressed_vocab_size: int = 99456  # ~23% reduction from 129280

    @staticmethod
    def get_name():
        return "deepseek-ai/Engram-27B"

    @classmethod
    def get_profiling_name(cls) -> str:
        return "microsoft/phi-2"


@dataclass
class MoE27BModelConfig(EngramBaseModelConfig):
    """MoE-27B: Pure MoE baseline (26.7B total params, no Engram).

    72 routed experts with 2 shared experts, top-6 activation.
    This is the iso-parameter baseline for Engram-27B.
    """
    is_moe: bool = True
    num_routed_experts: int = 72
    num_experts_per_tok: int = 6
    num_shared_experts: int = 2
    moe_intermediate_size: int = 2560

    @staticmethod
    def get_name():
        return "deepseek-ai/MoE-27B"

    @classmethod
    def get_profiling_name(cls) -> str:
        return "microsoft/phi-2"


@dataclass
class Engram40BModelConfig(EngramBaseModelConfig):
    """Engram-40B: Scaled-up Engram (39.5B total params).

    Same backbone as Engram-27B but with larger Engram memory
    (18.5B params) while keeping activated parameters fixed.
    """
    is_moe: bool = True
    num_routed_experts: int = 55
    num_experts_per_tok: int = 6
    num_shared_experts: int = 2
    moe_intermediate_size: int = 2560

    # Larger Engram memory
    has_engram: bool = True
    engram_layers: List = field(default_factory=lambda: [2, 15])
    engram_num_heads: int = 8
    engram_dim: int = 1280
    engram_ngram_sizes: List = field(default_factory=lambda: [2, 3])
    engram_total_params_b: float = 18.5
    engram_compressed_vocab_size: int = 99456

    @staticmethod
    def get_name():
        return "deepseek-ai/Engram-40B"

    @classmethod
    def get_profiling_name(cls) -> str:
        return "microsoft/phi-2"


@dataclass
class Qwen3_30B_A3BModelConfig(BaseModelConfig):
    """Qwen3-30B-A3B: MoE model with 128 routed experts, 8 active per token."""
    num_layers: int = 48
    num_q_heads: int = 32
    num_kv_heads: int = 4
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 11008  # dense fallback
    max_position_embeddings: int = 131072
    use_gated_mlp: bool = True
    use_bias: bool = False
    use_qkv_bias: bool = True
    activation: ActivationType = ActivationType.SILU
    norm: NormType = NormType.RMS_NORM
    post_attn_norm: bool = True
    vocab_size: int = 151936
    rope_theta: Optional[float] = 1000000

    # MoE
    is_moe: bool = True
    num_routed_experts: int = 128
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    moe_intermediate_size: int = 1408

    @staticmethod
    def get_name():
        return "Qwen/Qwen3-30B-A3B"

    @classmethod
    def get_profiling_name(cls) -> str:
        return "meta-llama/Meta-Llama-3-8B"


@dataclass
class Qwen3CoderNext80B_A3BModelConfig(BaseModelConfig):
    """Qwen3-Coder-Next 80B-A3B: Hybrid Transformer-Mamba MoE.

    80B total params, 3B active per token.  512 routed experts with
    10 active per token plus 1 shared expert.  Hybrid attention uses
    Gated DeltaNet (3/4 of layers) + Gated Attention (1/4 of layers);
    modeled here as GQA for profiling purposes.
    """
    num_layers: int = 48
    num_q_heads: int = 16
    num_kv_heads: int = 2
    embedding_dim: int = 2048
    mlp_hidden_dim: int = 5504  # dense fallback
    max_position_embeddings: int = 262144
    use_gated_mlp: bool = True
    use_bias: bool = False
    use_qkv_bias: bool = True
    activation: ActivationType = ActivationType.SILU
    norm: NormType = NormType.RMS_NORM
    post_attn_norm: bool = True
    vocab_size: int = 151936
    rope_theta: Optional[float] = 1000000

    # MoE: 512 routed experts, 10 active/token, 1 shared
    is_moe: bool = True
    num_routed_experts: int = 512
    num_experts_per_tok: int = 10
    num_shared_experts: int = 1
    moe_intermediate_size: int = 512

    @staticmethod
    def get_name():
        return "Qwen/Qwen3-Coder-Next-80B-A3B"

    @classmethod
    def get_profiling_name(cls) -> str:
        return "meta-llama/Meta-Llama-3-8B"


@dataclass
class Mixtral8x7BModelConfig(BaseModelConfig):
    """Mixtral-8x7B: MoE model with 8 routed experts, 2 active per token."""
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 8
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 14336
    max_position_embeddings: int = 32768
    use_gated_mlp: bool = True
    use_bias: bool = False
    use_qkv_bias: bool = False
    activation: ActivationType = ActivationType.SILU
    norm: NormType = NormType.RMS_NORM
    post_attn_norm: bool = True
    vocab_size: int = 32000
    rope_theta: Optional[float] = 1000000

    # MoE
    is_moe: bool = True
    num_routed_experts: int = 8
    num_experts_per_tok: int = 2
    num_shared_experts: int = 0
    moe_intermediate_size: int = 14336

    @staticmethod
    def get_name():
        return "mistralai/Mixtral-8x7B-v0.1"

    @classmethod
    def get_profiling_name(cls) -> str:
        return "meta-llama/Llama-2-7b-hf"
