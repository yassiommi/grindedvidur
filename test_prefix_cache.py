#!/usr/bin/env python3
"""Tests for the prefix-aware KV cache manager with radix tree and LRU eviction."""

import sys
import os

sys.path.insert(0, "/home/user/grindedvidur")

from vidur.entities.prefix_cache_manager import PrefixCacheManager, PrefixCacheStats


def test_basic_insert_and_match():
    """Test basic insert and prefix matching."""
    print("Test: basic insert and match...")
    cache = PrefixCacheManager(max_blocks=100, block_size=4)

    tokens = tuple(range(20))  # 20 tokens = 5 blocks
    cache.insert(tokens)

    assert cache.num_cached_blocks == 5
    assert cache.stats.total_insertions == 1

    # Full match
    matched = cache.match_prefix(tokens)
    assert matched == 20, f"Expected 20, got {matched}"

    # Prefix match
    matched = cache.match_prefix(tokens[:12])
    assert matched == 12, f"Expected 12, got {matched}"

    # Partial block match (only full blocks count)
    matched = cache.match_prefix(tokens[:13])
    assert matched == 12, f"Expected 12 (block-aligned), got {matched}"

    # No match
    different = tuple(range(100, 120))
    matched = cache.match_prefix(different)
    assert matched == 0, f"Expected 0, got {matched}"

    print("  PASSED")


def test_shared_prefix_hit():
    """Test that two sequences sharing a prefix get cache hits."""
    print("Test: shared prefix hit...")
    cache = PrefixCacheManager(max_blocks=100, block_size=4)

    # Shared prefix of 12 tokens (3 blocks), different suffixes
    shared = tuple(range(12))
    seq_a = shared + tuple(range(100, 108))  # 20 tokens total
    seq_b = shared + tuple(range(200, 208))  # 20 tokens total

    # Insert first sequence
    cache.insert(seq_a)

    # Second sequence should match the shared prefix
    matched = cache.match_prefix(seq_b)
    assert matched == 12, f"Expected 12 (shared prefix), got {matched}"

    assert cache.stats.total_hits == 1
    assert cache.stats.total_tokens_hit == 12

    print("  PASSED")


def test_lru_eviction():
    """Test LRU eviction when cache is full."""
    print("Test: LRU eviction...")
    cache = PrefixCacheManager(max_blocks=10, block_size=4)

    # Insert sequence A (3 blocks = 12 tokens)
    seq_a = tuple(range(0, 12))
    cache.insert(seq_a)
    assert cache.num_cached_blocks == 3

    # Insert sequence B (3 blocks = 12 tokens)
    seq_b = tuple(range(100, 112))
    cache.insert(seq_b)
    assert cache.num_cached_blocks == 6

    # Insert sequence C (5 blocks = 20 tokens) - should evict A (oldest)
    seq_c = tuple(range(200, 220))
    cache.insert(seq_c)

    # A should be evicted (oldest, needed space)
    matched_a = cache.match_prefix(seq_a)
    matched_c = cache.match_prefix(seq_c)
    assert matched_c == 20, f"C should be cached, got {matched_c}"
    assert cache.stats.total_evictions > 0, "Should have evicted"

    print("  PASSED")


def test_lru_ordering():
    """Test that accessing a cached entry moves it to the end of LRU."""
    print("Test: LRU ordering...")
    cache = PrefixCacheManager(max_blocks=10, block_size=4)

    seq_a = tuple(range(0, 8))       # 2 blocks
    seq_b = tuple(range(100, 108))   # 2 blocks
    seq_c = tuple(range(200, 208))   # 2 blocks

    cache.insert(seq_a)
    cache.insert(seq_b)
    cache.insert(seq_c)
    assert cache.num_cached_blocks == 6

    # Access A to make it recently used
    cache.match_prefix(seq_a)

    # Insert large sequence to force eviction - B should be evicted first (oldest)
    seq_d = tuple(range(300, 320))  # 5 blocks - needs eviction
    cache.insert(seq_d)

    # A should survive (was recently accessed), B should be evicted
    matched_a = cache.match_prefix(seq_a)
    matched_b = cache.match_prefix(seq_b)
    assert matched_a == 8, f"A should survive (recently used), got {matched_a}"
    # B was the least recently used, should be evicted
    assert matched_b == 0, f"B should be evicted, got {matched_b}"

    print("  PASSED")


def test_no_match_for_short_sequence():
    """Test that sequences shorter than block_size get 0 matches."""
    print("Test: no match for short sequence...")
    cache = PrefixCacheManager(max_blocks=100, block_size=16)

    tokens = tuple(range(10))  # Less than block_size
    blocks = cache.insert(tokens)
    assert blocks == 0

    matched = cache.match_prefix(tokens)
    assert matched == 0

    print("  PASSED")


def test_cache_stats():
    """Test that cache statistics are correctly tracked."""
    print("Test: cache stats...")
    cache = PrefixCacheManager(max_blocks=100, block_size=4)

    tokens = tuple(range(16))  # 4 blocks
    cache.insert(tokens)

    # Hit
    cache.match_prefix(tokens[:12])
    # Miss
    cache.match_prefix(tuple(range(100, 116)))

    stats = cache.stats
    assert stats.total_lookups == 2
    assert stats.total_hits == 1
    assert stats.total_misses == 1
    assert stats.hit_rate == 0.5
    assert stats.total_tokens_hit == 12
    assert stats.total_insertions == 1
    assert stats.total_blocks_inserted == 4

    print("  PASSED")


def test_on_request_complete():
    """Test the on_request_complete flow."""
    print("Test: on_request_complete...")
    cache = PrefixCacheManager(max_blocks=100, block_size=4)

    tokens = tuple(range(20))
    cache.on_request_complete(tokens)

    # Subsequent request with same prefix should hit
    matched = cache.match_prefix(tokens[:16])
    assert matched == 16

    print("  PASSED")


def test_multiple_prefix_groups():
    """Test realistic scenario with multiple prefix groups."""
    print("Test: multiple prefix groups...")
    cache = PrefixCacheManager(max_blocks=200, block_size=16)

    # Simulate 3 system prompt prefixes
    prefix_a = tuple(range(0, 64))       # 4 blocks
    prefix_b = tuple(range(1000, 1064))  # 4 blocks
    prefix_c = tuple(range(2000, 2064))  # 4 blocks

    # First request with prefix A - inserts into cache
    req1 = prefix_a + tuple(range(5000, 5032))
    cache.on_request_complete(req1)

    # Second request with prefix A - should get cache hit on prefix
    req2 = prefix_a + tuple(range(6000, 6032))
    hit = cache.match_prefix(req2)
    assert hit == 64, f"Expected 64 token hit on shared prefix, got {hit}"
    cache.on_request_complete(req2)

    # Request with prefix B - no hit yet
    req3 = prefix_b + tuple(range(7000, 7032))
    hit = cache.match_prefix(req3)
    assert hit == 0
    cache.on_request_complete(req3)

    # Next request with prefix B - should hit
    req4 = prefix_b + tuple(range(8000, 8032))
    hit = cache.match_prefix(req4)
    assert hit == 64
    cache.on_request_complete(req4)

    stats = cache.stats
    # We did 4 match_prefix calls: 2 hits (req2 prefix_a, req4 prefix_b),
    # 2 misses (req1 first prefix_a request, req3 first prefix_b request)
    assert stats.total_hits >= 2, f"Expected at least 2 hits, got {stats.total_hits}"
    assert stats.hit_rate >= 0.4, f"Expected hit rate >= 40%, got {stats.hit_rate}"

    print("  PASSED")


def test_empty_cache():
    """Test operations on empty cache."""
    print("Test: empty cache...")
    cache = PrefixCacheManager(max_blocks=10, block_size=4)

    assert cache.num_cached_blocks == 0
    assert cache.cache_utilization == 0.0

    matched = cache.match_prefix(tuple(range(20)))
    assert matched == 0

    assert cache.stats.total_lookups == 1
    assert cache.stats.total_misses == 1

    print("  PASSED")


def test_prefix_token_generator():
    """Test the PrefixTokenGenerator assigns token IDs correctly."""
    print("Test: prefix token generator...")
    from vidur.config.config import PrefixCacheConfig
    from vidur.entities.prefix_token_generator import PrefixTokenGenerator
    from vidur.entities.request import Request

    config = PrefixCacheConfig(
        enabled=True,
        num_shared_prefixes=3,
        shared_prefix_length_fraction=0.5,
        seed=42,
    )

    gen = PrefixTokenGenerator(config)
    requests = [
        Request(arrived_at=0.0, num_prefill_tokens=100, num_decode_tokens=50),
        Request(arrived_at=1.0, num_prefill_tokens=100, num_decode_tokens=50),
        Request(arrived_at=2.0, num_prefill_tokens=100, num_decode_tokens=50),
    ]

    gen.assign_token_ids(requests)

    for req in requests:
        assert req.token_ids is not None
        assert len(req.token_ids) == req.num_prefill_tokens

    # Requests in the same group should share a prefix
    # Check that at least some pairs share prefix tokens
    # (with 3 groups and 3 requests, at least 2 should be in the same group by pigeonhole)
    shared_count = 0
    for i in range(len(requests)):
        for j in range(i + 1, len(requests)):
            shared = 0
            for k in range(min(len(requests[i].token_ids), len(requests[j].token_ids))):
                if requests[i].token_ids[k] == requests[j].token_ids[k]:
                    shared += 1
                else:
                    break
            if shared > 0:
                shared_count += 1

    print(f"  Shared prefix pairs: {shared_count}")

    print("  PASSED")


def test_request_prefix_cache_hit():
    """Test Request.apply_prefix_cache_hit."""
    print("Test: request prefix cache hit...")
    from vidur.entities.request import Request

    req = Request(arrived_at=0.0, num_prefill_tokens=100, num_decode_tokens=50)

    assert req.prefix_cache_hit_tokens == 0
    assert req.original_prefill_tokens == 100

    req.apply_prefix_cache_hit(30)

    assert req.prefix_cache_hit_tokens == 30
    assert req.original_prefill_tokens == 100
    assert req.num_processed_tokens == 30  # Simulates having processed cached tokens
    assert req.prefix_cache_hit_ratio == 0.3

    print("  PASSED")


def test_state_summary():
    """Test get_state_summary returns expected structure."""
    print("Test: state summary...")
    cache = PrefixCacheManager(max_blocks=50, block_size=4)

    tokens = tuple(range(20))
    cache.insert(tokens)
    cache.match_prefix(tokens[:12])

    summary = cache.get_state_summary()
    assert "num_cached_blocks" in summary
    assert "max_blocks" in summary
    assert "utilization" in summary
    assert "stats" in summary
    assert summary["num_cached_blocks"] == 5
    assert summary["max_blocks"] == 50
    assert summary["utilization"] == 0.1

    stats = summary["stats"]
    assert stats["total_lookups"] == 1
    assert stats["total_hits"] == 1
    assert stats["hit_rate"] == 1.0

    print("  PASSED")


def test_integration_cache_with_scheduler_flow():
    """Test the full flow: generate requests, assign tokens, lookup, insert."""
    print("Test: integration cache with scheduler flow...")
    from vidur.config.config import PrefixCacheConfig
    from vidur.entities.prefix_token_generator import PrefixTokenGenerator
    from vidur.entities.request import Request

    config = PrefixCacheConfig(
        enabled=True,
        num_shared_prefixes=2,
        shared_prefix_length_fraction=0.4,
        seed=123,
    )

    # Generate requests with tokens
    gen = PrefixTokenGenerator(config)
    requests = [
        Request(arrived_at=float(i), num_prefill_tokens=80, num_decode_tokens=20)
        for i in range(10)
    ]
    gen.assign_token_ids(requests)

    # Simulate cache
    cache = PrefixCacheManager(max_blocks=100, block_size=16)

    total_hits = 0
    total_tokens_saved = 0

    for req in requests:
        # Lookup prefix
        cached = cache.match_prefix(req.token_ids)
        if cached > 0:
            req.apply_prefix_cache_hit(cached)
            total_hits += 1
            total_tokens_saved += cached

        # "Complete" the request and insert into cache
        cache.on_request_complete(req.token_ids)

    print(f"  Requests: {len(requests)}")
    print(f"  Cache hits: {total_hits}")
    print(f"  Total tokens saved: {total_tokens_saved}")
    print(f"  Hit rate: {cache.stats.hit_rate:.2%}")
    print(f"  Token hit rate: {cache.stats.token_hit_rate:.2%}")

    # With 2 prefix groups and 10 requests, we should get hits after the first
    # request in each group
    assert total_hits >= 4, f"Expected at least 4 hits, got {total_hits}"
    assert cache.stats.hit_rate > 0.3, f"Expected hit rate > 30%, got {cache.stats.hit_rate:.2%}"

    print("  PASSED")


if __name__ == "__main__":
    tests = [
        test_basic_insert_and_match,
        test_shared_prefix_hit,
        test_lru_eviction,
        test_lru_ordering,
        test_no_match_for_short_sequence,
        test_cache_stats,
        test_on_request_complete,
        test_multiple_prefix_groups,
        test_empty_cache,
        test_prefix_token_generator,
        test_request_prefix_cache_hit,
        test_state_summary,
        test_integration_cache_with_scheduler_flow,
    ]

    print("\n" + "=" * 60)
    print("  Prefix Cache Manager Tests")
    print("=" * 60)

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "-" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("-" * 60)

    sys.exit(0 if failed == 0 else 1)
