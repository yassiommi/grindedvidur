import atexit
import heapq
import json
from typing import List

from vidur.config import SimulationConfig
from vidur.entities import Cluster
from vidur.entities.prefix_token_generator import PrefixTokenGenerator
from vidur.events import BaseEvent, RequestArrivalEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.request_generator import RequestGeneratorRegistry
from vidur.scheduler import BaseGlobalScheduler, GlobalSchedulerRegistry

logger = init_logger(__name__)


class Simulator:
    def __init__(self, config: SimulationConfig) -> None:
        self._config: SimulationConfig = config

        self._time = 0
        self._terminate = False
        self._time_limit = self._config.time_limit
        if not self._time_limit:
            self._time_limit = float("inf")

        self._event_queue = []

        self._event_trace = []
        self._event_chrome_trace = []

        self._cluster = Cluster(
            self._config.cluster_config,
            self._config.metrics_config,
            self._config.request_generator_config,
        )
        self._metric_store = MetricsStore(self._config)
        self._request_generator = RequestGeneratorRegistry.get(
            self._config.request_generator_config.get_type(),
            self._config.request_generator_config,
        )
        self._scheduler = GlobalSchedulerRegistry.get(
            self._config.cluster_config.global_scheduler_config.get_type(),
            self._config,
            self._cluster.replicas,
        )

        self._init_event_queue()
        atexit.register(self._write_output)

    @property
    def scheduler(self) -> BaseGlobalScheduler:
        return self._scheduler

    @property
    def metric_store(self) -> MetricsStore:
        return self._metric_store

    def run(self) -> None:
        logger.info(
            f"Starting simulation with cluster: {self._cluster} and {len(self._event_queue)} requests"
        )

        while self._event_queue and not self._terminate:
            _, event = heapq.heappop(self._event_queue)
            self._set_time(event._time)
            new_events = event.handle_event(self._scheduler, self._metric_store)
            self._add_events(new_events)

            if self._config.metrics_config.write_json_trace:
                self._event_trace.append(event.to_dict())

            if self._config.metrics_config.enable_chrome_trace:
                chrome_trace = event.to_chrome_trace()
                if chrome_trace:
                    self._event_chrome_trace.append(chrome_trace)

        assert self._scheduler.is_empty() or self._terminate

        logger.info(f"Simulation ended at: {self._time}s")

    def _write_output(self) -> None:
        logger.info("Writing output")

        # Log prefix cache summary if enabled
        self._write_prefix_cache_summary()

        self._metric_store.plot()
        logger.info("Metrics written")

        if self._config.metrics_config.write_json_trace:
            self._write_event_trace()
            logger.info("Json event trace written")

        if self._config.metrics_config.enable_chrome_trace:
            self._write_chrome_trace()
            logger.info("Chrome event trace written")

    def _add_event(self, event: BaseEvent) -> None:
        heapq.heappush(self._event_queue, (event._priority_number, event))

    def _add_events(self, events: List[BaseEvent]) -> None:
        for event in events:
            self._add_event(event)

    def _init_event_queue(self) -> None:
        requests = self._request_generator.generate()

        # Assign synthetic token IDs if prefix caching is enabled
        prefix_cache_config = (
            self._config.cluster_config.replica_scheduler_config.prefix_cache_config
        )
        if prefix_cache_config.enabled:
            token_generator = PrefixTokenGenerator(prefix_cache_config)
            token_generator.assign_token_ids(requests)
            logger.info(
                f"Assigned synthetic token IDs to {len(requests)} requests "
                f"({prefix_cache_config.num_shared_prefixes} prefix groups, "
                f"{prefix_cache_config.shared_prefix_length_fraction:.0%} shared)"
            )

        for request in requests:
            self._add_event(RequestArrivalEvent(request.arrived_at, request))

    def _set_time(self, time: float) -> None:
        self._time = time
        if self._time > self._time_limit:
            logger.info(
                f"Time limit reached: {self._time_limit}s terminating the simulation."
            )
            self._terminate = True

    def _write_prefix_cache_summary(self) -> None:
        """Write prefix cache statistics summary if enabled."""
        for replica_scheduler in self._scheduler.replica_schedulers:
            cache = replica_scheduler.prefix_cache
            if cache is None:
                continue
            summary = cache.get_state_summary()
            stats = summary["stats"]
            logger.info(
                f"Prefix cache stats (replica {replica_scheduler.replica_id}): "
                f"hit_rate={stats['hit_rate']:.2%}, "
                f"token_hit_rate={stats['token_hit_rate']:.2%}, "
                f"lookups={stats['total_lookups']}, "
                f"hits={stats['total_hits']}, "
                f"evictions={stats['total_evictions']}, "
                f"utilization={summary['utilization']:.2%}"
            )
            # Write to file
            output_path = (
                f"{self._config.metrics_config.output_dir}"
                f"/prefix_cache_stats_replica_{replica_scheduler.replica_id}.json"
            )
            import json as json_mod
            with open(output_path, "w") as f:
                json_mod.dump(summary, f, indent=2)

    def _write_event_trace(self) -> None:
        trace_file = f"{self._config.metrics_config.output_dir}/event_trace.json"
        with open(trace_file, "w") as f:
            json.dump(self._event_trace, f)

    def _write_chrome_trace(self) -> None:
        trace_file = f"{self._config.metrics_config.output_dir}/chrome_trace.json"

        chrome_trace = {"traceEvents": self._event_chrome_trace}

        with open(trace_file, "w") as f:
            json.dump(chrome_trace, f)
