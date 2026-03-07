from dataclasses import dataclass, field


@dataclass
class KvCacheMetadata:
    """Tracks KV cache properties and transfer metadata for a request."""

    request_id: int
    num_tokens: int
    """Number of tokens that generated the KV cache (prefill tokens)."""

    bytes_per_token: float
    """KV cache bytes per token."""

    cache_bytes: float = field(init=False)
    """Total KV cache bytes for this request."""

    source_gpu: int = 0
    """Source GPU ID (prefill GPU)."""

    dest_gpu: int = 1
    """Destination GPU ID (decode GPU)."""

    transfer_time_ms: float = 0.0
    """Transfer latency in milliseconds."""

    transfer_throughput_gbps: float = 0.0
    """Effective transfer throughput in GB/s."""

    prefetch_overlap_ms: float = 0.0
    """Time overlapped with concurrent decode computation in milliseconds."""

    transferred_at: float = 0.0
    """Timestamp when transfer completed."""

    def __post_init__(self):
        self.cache_bytes = self.num_tokens * self.bytes_per_token

    @property
    def effective_transfer_latency_ms(self) -> float:
        """Transfer latency minus any prefetch overlap savings."""
        return max(0.0, self.transfer_time_ms - self.prefetch_overlap_ms)
