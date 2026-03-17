from abc import ABC, abstractmethod
from math import ceil
from typing import List, Optional

from vidur.config import (
    BaseReplicaSchedulerConfig,
    BaseRequestGeneratorConfig,
    ReplicaConfig,
)
from vidur.entities import Batch, Replica, Request
from vidur.entities.prefix_cache_manager import PrefixCacheManager
from vidur.execution_time_predictor import BaseExecutionTimePredictor
from vidur.logger import init_logger
from vidur.scheduler.replica_stage_scheduler import ReplicaStageScheduler
from vidur.scheduler.utils.memory_planner import MemoryPlanner

logger = init_logger(__name__)


class BaseReplicaScheduler(ABC):
    def __init__(
        self,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        request_generator_config: BaseRequestGeneratorConfig,
        replica: Replica,
        num_stages: int,
        execution_time_predictor: BaseExecutionTimePredictor,
    ) -> None:
        self._config = replica_scheduler_config
        self._replica_config = replica_config
        self._request_generator_config = request_generator_config
        self._replica_id = replica.id
        self._num_stages = num_stages

        self._max_blocks_per_sequence = (
            self._request_generator_config.max_tokens // self._config.block_size
        )

        memory_planner = MemoryPlanner(self._replica_config, replica)

        if not self._config.num_blocks:
            self._config.num_blocks = (
                self._max_blocks_per_sequence * memory_planner.get_max_request_slots()
            )
        self._max_batch_size = min(
            memory_planner.get_max_batch_size(),
            self._config.batch_size_cap,
        )

        logger.debug(
            f"Obtained max batch size of {self._max_batch_size} for replica {self._replica_id}"
        )

        self._request_queue = []
        self._num_allocated_blocks = 0
        self._allocation_map = {}

        # Initialize prefix cache if enabled
        self._prefix_cache: Optional[PrefixCacheManager] = None
        prefix_cache_config = self._config.prefix_cache_config
        if prefix_cache_config.enabled:
            prefix_cache_blocks = int(
                self._config.num_blocks * prefix_cache_config.max_blocks_fraction
            )
            self._prefix_cache = PrefixCacheManager(
                max_blocks=prefix_cache_blocks,
                block_size=self._config.block_size,
            )
            logger.info(
                f"Prefix cache enabled for replica {self._replica_id}: "
                f"{prefix_cache_blocks} blocks "
                f"({prefix_cache_config.max_blocks_fraction:.0%} of {self._config.num_blocks})"
            )

        self._replica_stage_schedulers = {
            stage_id: ReplicaStageScheduler(
                replica.id,
                stage_id,
                stage_id == num_stages - 1,
                execution_time_predictor,
            )
            for stage_id in range(num_stages)
        }

    @property
    def num_pending_requests(self) -> int:
        return len(self._request_queue)

    @property
    def replica_id(self) -> int:
        return self._replica_id

    @property
    def num_allocated_blocks(self) -> int:
        return self._num_allocated_blocks

    @property
    def memory_usage_percent(self) -> int:
        return (self._num_allocated_blocks * 100) / self._config.num_blocks

    def is_empty(self) -> bool:
        return (
            self.num_pending_requests == 0
            and len(self._allocation_map) == 0
            and all(
                stage_scheduler.is_empty()
                for stage_scheduler in self._replica_stage_schedulers.values()
            )
        )

    def _get_request_next_num_tokens(self, request: Request) -> int:
        assert not request.completed

        if request.is_prefill_complete:
            return 1

        # Account for prefix cache hits: only prefill uncached tokens
        remaining_prefill = request.num_prefill_tokens - request.num_processed_tokens
        return max(1, remaining_prefill)

    @property
    def prefix_cache(self) -> Optional[PrefixCacheManager]:
        return self._prefix_cache

    def add_request(self, request: Request) -> None:
        # Perform prefix cache lookup if enabled and request has token IDs
        if self._prefix_cache is not None and request.token_ids is not None:
            cached_tokens = self._prefix_cache.match_prefix(request.token_ids)
            if cached_tokens > 0:
                request.apply_prefix_cache_hit(cached_tokens)
                logger.debug(
                    f"Request {request.id}: prefix cache hit for {cached_tokens} tokens "
                    f"(prefill reduced from {request.original_prefill_tokens} to "
                    f"{request.num_prefill_tokens - request.num_processed_tokens})"
                )
        self._request_queue.append(request)

    def get_replica_stage_scheduler(self, stage_id: int):
        return self._replica_stage_schedulers[stage_id]

    def can_allocate(self, num_blocks: int) -> bool:
        return self._config.num_blocks - self._num_allocated_blocks >= num_blocks

    def allocate(self, request_id: int, num_blocks: int) -> None:
        self._num_allocated_blocks += num_blocks
        if request_id not in self._allocation_map:
            self._allocation_map[request_id] = num_blocks
        else:
            self._allocation_map[request_id] += num_blocks

        assert self._num_allocated_blocks <= self._config.num_blocks

    def free(self, *request_ids: List[int]) -> None:
        for request_id in request_ids:
            num_blocks = self._allocation_map.pop(request_id)
            self._num_allocated_blocks -= num_blocks

        assert self._num_allocated_blocks >= 0

    def free_batch(self, batch: Batch) -> None:
        self.free(*batch.request_ids)

    def _insert_completed_into_prefix_cache(self, batch: Batch) -> None:
        """Insert completed requests' token sequences into the prefix cache."""
        if self._prefix_cache is None:
            return
        for request in batch.requests:
            if request.completed and request.token_ids is not None:
                self._prefix_cache.on_request_complete(request.token_ids)

    @abstractmethod
    def on_batch_end(self, batch: Batch) -> None:
        pass

    @abstractmethod
    def _get_next_batch(self) -> Batch:
        pass

    def on_schedule(self) -> List[Batch]:
        scheduled_batches = []
        while self._num_running_batches < self._num_stages:
            batch = self._get_next_batch()
            if not batch:
                break
            scheduled_batches.append(batch)
            self._num_running_batches += 1
        return scheduled_batches
