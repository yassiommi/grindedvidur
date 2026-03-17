"""Prefix-aware KV cache manager using a radix tree with LRU eviction.

Models SGLang-style prefix caching where requests sharing a common token
prefix can reuse cached KV blocks, avoiding redundant prefill computation.

The radix tree maps token sequences to cached block ranges. LRU eviction
operates on leaf nodes when the cache is full. When a new request arrives,
the tree is walked to find the longest cached prefix (a "hit"), which
reduces the number of tokens that need prefilling.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from vidur.logger import init_logger

logger = init_logger(__name__)


@dataclass
class PrefixCacheStats:
    """Accumulated statistics for prefix cache operations."""

    total_lookups: int = 0
    total_hits: int = 0  # lookups where at least 1 token was cached
    total_misses: int = 0  # lookups where 0 tokens were cached
    total_tokens_looked_up: int = 0
    total_tokens_hit: int = 0  # tokens found in cache (avoided prefill)
    total_tokens_missed: int = 0  # tokens not in cache (need prefill)
    total_evictions: int = 0  # number of eviction events
    total_blocks_evicted: int = 0
    total_insertions: int = 0
    total_blocks_inserted: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_lookups == 0:
            return 0.0
        return self.total_hits / self.total_lookups

    @property
    def token_hit_rate(self) -> float:
        if self.total_tokens_looked_up == 0:
            return 0.0
        return self.total_tokens_hit / self.total_tokens_looked_up

    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate

    def to_dict(self) -> dict:
        return {
            "total_lookups": self.total_lookups,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": self.hit_rate,
            "token_hit_rate": self.token_hit_rate,
            "total_tokens_looked_up": self.total_tokens_looked_up,
            "total_tokens_hit": self.total_tokens_hit,
            "total_tokens_missed": self.total_tokens_missed,
            "total_evictions": self.total_evictions,
            "total_blocks_evicted": self.total_blocks_evicted,
            "total_insertions": self.total_insertions,
            "total_blocks_inserted": self.total_blocks_inserted,
        }


class RadixTreeNode:
    """A node in the radix tree representing a sequence of token blocks.

    Each node stores a segment of token IDs and has children keyed by the
    first token ID of the child's segment. Leaf nodes (or any node with
    cached blocks) participate in LRU tracking.
    """

    __slots__ = ["token_segment", "children", "num_blocks", "parent", "first_token"]

    def __init__(
        self,
        token_segment: Tuple[int, ...],
        parent: Optional["RadixTreeNode"] = None,
    ):
        self.token_segment = token_segment
        self.children: Dict[int, "RadixTreeNode"] = {}
        self.num_blocks = 0  # number of KV cache blocks held by this node
        self.parent = parent
        self.first_token = token_segment[0] if token_segment else -1

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def __repr__(self) -> str:
        seg = self.token_segment[:3]
        return f"RadixTreeNode(seg={seg}..., blocks={self.num_blocks}, children={len(self.children)})"


class PrefixCacheManager:
    """Manages a prefix-aware KV cache using a radix tree with LRU eviction.

    The cache operates at block granularity (block_size tokens per block).
    Token sequences are inserted into a radix tree, and the longest matching
    prefix is returned on lookup. LRU eviction removes the least recently
    used leaf nodes when the cache exceeds capacity.

    Args:
        max_blocks: Maximum number of KV cache blocks the cache can hold.
        block_size: Number of tokens per block.
    """

    def __init__(self, max_blocks: int, block_size: int):
        self._max_blocks = max_blocks
        self._block_size = block_size
        self._num_cached_blocks = 0

        # Root node of the radix tree (empty segment)
        self._root = RadixTreeNode(token_segment=())

        # LRU tracking: maps node id -> node, ordered by access time (oldest first)
        self._lru_order: OrderedDict[int, RadixTreeNode] = OrderedDict()

        self._stats = PrefixCacheStats()

    @property
    def stats(self) -> PrefixCacheStats:
        return self._stats

    @property
    def num_cached_blocks(self) -> int:
        return self._num_cached_blocks

    @property
    def max_blocks(self) -> int:
        return self._max_blocks

    @property
    def cache_utilization(self) -> float:
        if self._max_blocks == 0:
            return 0.0
        return self._num_cached_blocks / self._max_blocks

    def _tokens_to_blocks(self, num_tokens: int) -> int:
        """Convert token count to block count (floor division - only full blocks cached)."""
        return num_tokens // self._block_size

    def match_prefix(self, token_ids: Tuple[int, ...]) -> int:
        """Find the longest cached prefix for a token sequence.

        Walks the radix tree matching token_ids and returns the number of
        tokens that are already cached (aligned to block boundaries).

        Args:
            token_ids: The full token sequence of the request.

        Returns:
            Number of tokens with cached KV state (block-aligned).
        """
        self._stats.total_lookups += 1
        self._stats.total_tokens_looked_up += len(token_ids)

        matched_tokens = 0
        node = self._root
        pos = 0

        while pos < len(token_ids):
            next_token = token_ids[pos]
            if next_token not in node.children:
                break

            child = node.children[next_token]
            seg = child.token_segment
            seg_len = len(seg)

            # Check how much of this node's segment matches
            match_len = 0
            while match_len < seg_len and pos + match_len < len(token_ids):
                if token_ids[pos + match_len] != seg[match_len]:
                    break
                match_len += 1

            if match_len == 0:
                break

            # Count the blocks covered by matched portion
            matched_tokens += match_len
            pos += match_len

            # Touch this node in LRU (mark as recently used)
            node_id = id(child)
            if node_id in self._lru_order:
                self._lru_order.move_to_end(node_id)

            if match_len < seg_len:
                # Partial match within a node - can't go deeper
                break

            node = child

        # Align to block boundary (only full blocks count as cached)
        cached_tokens = (matched_tokens // self._block_size) * self._block_size

        if cached_tokens > 0:
            self._stats.total_hits += 1
            self._stats.total_tokens_hit += cached_tokens
        else:
            self._stats.total_misses += 1

        self._stats.total_tokens_missed += len(token_ids) - cached_tokens

        return cached_tokens

    def insert(self, token_ids: Tuple[int, ...]) -> int:
        """Insert a token sequence into the cache, evicting if necessary.

        Only full blocks are cached. Tokens beyond the last full block
        boundary are not inserted.

        Args:
            token_ids: The full token sequence to cache.

        Returns:
            Number of new blocks inserted.
        """
        if len(token_ids) < self._block_size:
            return 0

        # Align to block boundary
        num_cacheable_tokens = (len(token_ids) // self._block_size) * self._block_size
        token_ids = token_ids[:num_cacheable_tokens]

        # Walk tree to find where new tokens diverge from existing cache
        node = self._root
        pos = 0

        while pos < len(token_ids):
            next_token = token_ids[pos]
            if next_token not in node.children:
                break

            child = node.children[next_token]
            seg = child.token_segment
            seg_len = len(seg)

            # Check match length
            match_len = 0
            while match_len < seg_len and pos + match_len < len(token_ids):
                if token_ids[pos + match_len] != seg[match_len]:
                    break
                match_len += 1

            if match_len == 0:
                break

            # Touch in LRU
            node_id = id(child)
            if node_id in self._lru_order:
                self._lru_order.move_to_end(node_id)

            if match_len < seg_len:
                # Partial match - need to split this node
                self._split_node(child, match_len)
                pos += match_len
                node = node.children[token_ids[pos - match_len]]
                break

            pos += seg_len
            node = child

        # Everything from pos onward is new and needs to be inserted
        if pos >= len(token_ids):
            return 0  # entire sequence already cached

        new_tokens = token_ids[pos:]
        new_blocks = self._tokens_to_blocks(len(new_tokens))

        if new_blocks == 0:
            return 0

        # Align new_tokens to block boundary
        new_tokens = new_tokens[: new_blocks * self._block_size]

        # Evict if necessary to make room
        self._evict_if_needed(new_blocks)

        # Insert new node
        new_node = RadixTreeNode(token_segment=tuple(new_tokens), parent=node)
        new_node.num_blocks = new_blocks
        node.children[new_tokens[0]] = new_node

        self._num_cached_blocks += new_blocks
        self._lru_order[id(new_node)] = new_node

        self._stats.total_insertions += 1
        self._stats.total_blocks_inserted += new_blocks

        return new_blocks

    def _split_node(self, node: RadixTreeNode, split_pos: int) -> None:
        """Split a node at the given position within its token segment.

        The original node keeps tokens[:split_pos] and a new child gets
        tokens[split_pos:] along with the original children.
        """
        prefix_seg = node.token_segment[:split_pos]
        suffix_seg = node.token_segment[split_pos:]

        # Create suffix child with original's children and blocks
        suffix_node = RadixTreeNode(token_segment=suffix_seg, parent=node)
        suffix_node.children = node.children
        suffix_node.num_blocks = node.num_blocks

        # Update parent references of moved children
        for child in suffix_node.children.values():
            child.parent = suffix_node

        # Update LRU: remove old, add suffix
        old_id = id(node)
        if old_id in self._lru_order:
            self._lru_order.pop(old_id)
        self._lru_order[id(suffix_node)] = suffix_node

        # Update original node to be the prefix
        node.token_segment = prefix_seg
        node.first_token = prefix_seg[0] if prefix_seg else -1
        node.children = {suffix_seg[0]: suffix_node}
        node.num_blocks = self._tokens_to_blocks(len(prefix_seg))

        # Adjust block counts
        suffix_node.num_blocks = (
            self._tokens_to_blocks(len(prefix_seg) + len(suffix_seg))
            - node.num_blocks
        )

        # Add prefix node to LRU if it has blocks
        if node.num_blocks > 0:
            self._lru_order[id(node)] = node

    def _evict_if_needed(self, blocks_needed: int) -> None:
        """Evict least-recently-used leaf nodes until enough space is available."""
        while (
            self._num_cached_blocks + blocks_needed > self._max_blocks
            and self._lru_order
        ):
            # Pop the oldest (least recently used) entry
            node_id, node = self._lru_order.popitem(last=False)

            # Only evict leaf nodes; if not a leaf, skip and re-add at end
            # (this ensures we only evict nodes that aren't shared prefixes
            # of active sequences)
            if not node.is_leaf:
                self._lru_order[node_id] = node
                # If we can't evict anything, break to avoid infinite loop
                # This means the cache is full of shared prefixes
                break

            self._num_cached_blocks -= node.num_blocks
            self._stats.total_evictions += 1
            self._stats.total_blocks_evicted += node.num_blocks

            # Remove from parent
            if node.parent is not None:
                parent = node.parent
                keys_to_remove = [
                    k for k, v in parent.children.items() if v is node
                ]
                for k in keys_to_remove:
                    del parent.children[k]

                # If parent now has exactly one child, merge them
                if len(parent.children) == 1 and parent.parent is not None:
                    self._merge_with_child(parent)

            logger.debug(
                f"Evicted {node.num_blocks} blocks from prefix cache "
                f"(cached: {self._num_cached_blocks}/{self._max_blocks})"
            )

    def _merge_with_child(self, node: RadixTreeNode) -> None:
        """Merge a node with its single child to maintain radix tree compactness."""
        if len(node.children) != 1 or node.parent is None:
            return

        child = next(iter(node.children.values()))

        # Merge segments
        merged_segment = node.token_segment + child.token_segment
        merged_blocks = node.num_blocks + child.num_blocks

        # Remove old entries from LRU
        old_node_id = id(node)
        old_child_id = id(child)
        if old_node_id in self._lru_order:
            self._lru_order.pop(old_node_id)
        if old_child_id in self._lru_order:
            self._lru_order.pop(old_child_id)

        # Update node in place
        node.token_segment = merged_segment
        node.first_token = merged_segment[0] if merged_segment else -1
        node.num_blocks = merged_blocks
        node.children = child.children

        # Update grandchildren's parent references
        for grandchild in node.children.values():
            grandchild.parent = node

        # Add merged node to LRU
        self._lru_order[id(node)] = node

    def on_request_complete(self, token_ids: Tuple[int, ...]) -> None:
        """Called when a request completes. Inserts its full token sequence.

        This models the behavior where completed requests' KV cache entries
        become available for prefix sharing with future requests.

        Args:
            token_ids: The full token sequence of the completed request.
        """
        self.insert(token_ids)

    def get_state_summary(self) -> dict:
        """Return a summary of the cache state for debugging/logging."""
        return {
            "num_cached_blocks": self._num_cached_blocks,
            "max_blocks": self._max_blocks,
            "utilization": self.cache_utilization,
            "stats": self._stats.to_dict(),
        }
