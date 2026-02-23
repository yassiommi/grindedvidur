from sklearn.ensemble import RandomForestRegressor

from vidur.config import (
    BaseReplicaSchedulerConfig,
    MetricsConfig,
    RandomForrestExecutionTimePredictorConfig,
    ReplicaConfig,
)
from vidur.execution_time_predictor.moe_execution_time_predictor import (
    MoEExecutionTimePredictor,
)


class RandomForrestExecutionTimePredictor(MoEExecutionTimePredictor):
    """Random forest predictor with automatic MoE support.

    Inherits from MoEExecutionTimePredictor (which extends SklearnExecutionTimePredictor).
    MoE methods only activate when the model config has is_moe=True, so dense
    models are unaffected.
    """

    def __init__(
        self,
        predictor_config: RandomForrestExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        metrics_config: MetricsConfig,
    ) -> None:
        # will trigger model training + MoE setup if applicable
        super().__init__(
            predictor_config=predictor_config,
            replica_config=replica_config,
            replica_scheduler_config=replica_scheduler_config,
            metrics_config=metrics_config,
        )

    def _get_grid_search_params(self):
        return {
            "n_estimators": self._config.num_estimators,
            "max_depth": self._config.max_depth,
            "min_samples_split": self._config.min_samples_split,
        }

    def _get_estimator(self):
        return RandomForestRegressor()
