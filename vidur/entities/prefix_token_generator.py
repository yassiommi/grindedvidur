"""Generates synthetic token IDs for prefix cache simulation.

Since Vidur's request generators only produce token counts (not actual token
sequences), this module generates synthetic token IDs that model realistic
prefix-sharing patterns. Requests are assigned to prefix groups, where all
requests in a group share a common system prompt prefix followed by unique
per-request tokens.

This enables the PrefixCacheManager to simulate cache hits for shared prefixes
without requiring actual tokenized text data.
"""

import random
from typing import List, Tuple

from vidur.config.config import PrefixCacheConfig
from vidur.entities.request import Request
from vidur.logger import init_logger

logger = init_logger(__name__)

# Token ID ranges to avoid collisions between prefix and unique tokens
_PREFIX_TOKEN_BASE = 1_000_000
_UNIQUE_TOKEN_BASE = 2_000_000


class PrefixTokenGenerator:
    """Assigns synthetic token IDs to requests with configurable prefix sharing.

    Each prefix group has a fixed shared prefix (simulating a system prompt).
    Requests are assigned to groups round-robin (or randomly), and their token
    IDs consist of [shared_prefix_tokens | unique_per_request_tokens].

    Args:
        config: PrefixCacheConfig with sharing parameters.
    """

    def __init__(self, config: PrefixCacheConfig):
        self._config = config
        self._rng = random.Random(config.seed)
        self._shared_prefixes: List[Tuple[int, ...]] = []
        self._unique_counter = 0

    def _ensure_prefixes_generated(self, max_prefix_length: int) -> None:
        """Generate shared prefix token sequences lazily."""
        if self._shared_prefixes:
            return

        for group_idx in range(self._config.num_shared_prefixes):
            prefix_tokens = tuple(
                _PREFIX_TOKEN_BASE + group_idx * max_prefix_length + i
                for i in range(max_prefix_length)
            )
            self._shared_prefixes.append(prefix_tokens)

        logger.debug(
            f"Generated {len(self._shared_prefixes)} shared prefix groups "
            f"(max length: {max_prefix_length} tokens)"
        )

    def assign_token_ids(self, requests: List[Request]) -> None:
        """Assign synthetic token IDs to a list of requests.

        Each request gets token_ids = shared_prefix + unique_suffix, where
        the shared prefix length is determined by shared_prefix_length_fraction
        of the request's prefill tokens.

        Args:
            requests: List of Request objects to assign token IDs to.
        """
        if not requests:
            return

        max_prefill = max(r.num_prefill_tokens for r in requests)
        max_prefix_len = max(
            1, int(max_prefill * self._config.shared_prefix_length_fraction)
        )
        self._ensure_prefixes_generated(max_prefix_len)

        for request in requests:
            num_prefill = request.num_prefill_tokens
            shared_len = int(num_prefill * self._config.shared_prefix_length_fraction)
            shared_len = max(0, shared_len)

            # Assign to a random prefix group
            group_idx = self._rng.randint(0, self._config.num_shared_prefixes - 1)
            shared_prefix = self._shared_prefixes[group_idx][:shared_len]

            # Generate unique tokens for the remainder
            unique_len = num_prefill - shared_len
            unique_tokens = tuple(
                _UNIQUE_TOKEN_BASE + self._unique_counter + i
                for i in range(unique_len)
            )
            self._unique_counter += unique_len

            token_ids = shared_prefix + unique_tokens
            request._token_ids = token_ids
