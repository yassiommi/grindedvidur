from dataclasses import dataclass, field

from vidur.config.base_fixed_config import BaseFixedConfig
from vidur.logger import init_logger
from vidur.types import DeviceSKUType, NodeSKUType

logger = init_logger(__name__)


@dataclass
class BaseNodeSKUConfig(BaseFixedConfig):
    num_devices_per_node: int
    # Intra-node interconnect bandwidth in GB/s (unidirectional, e.g. NVLink)
    intra_node_bw_gb_per_s: float = 0.0


@dataclass
class A40PairwiseNvlinkNodeSKUConfig(BaseNodeSKUConfig):
    device_sku_type: DeviceSKUType = DeviceSKUType.A40
    num_devices_per_node: int = 8
    intra_node_bw_gb_per_s: float = 100.0  # NVLink pairwise

    @staticmethod
    def get_type():
        return NodeSKUType.A40_PAIRWISE_NVLINK


@dataclass
class A100PairwiseNvlinkNodeSKUConfig(BaseNodeSKUConfig):
    device_sku_type: DeviceSKUType = DeviceSKUType.A100
    num_devices_per_node: int = 4
    intra_node_bw_gb_per_s: float = 300.0  # NVLink 3.0 pairwise

    @staticmethod
    def get_type():
        return NodeSKUType.A100_PAIRWISE_NVLINK


@dataclass
class H100PairwiseNvlinkNodeSKUConfig(BaseNodeSKUConfig):
    device_sku_type: DeviceSKUType = DeviceSKUType.H100
    num_devices_per_node: int = 4
    intra_node_bw_gb_per_s: float = 450.0  # NVLink 4.0 pairwise

    @staticmethod
    def get_type():
        return NodeSKUType.H100_PAIRWISE_NVLINK


@dataclass
class A100DgxNodeSKUConfig(BaseNodeSKUConfig):
    device_sku_type: DeviceSKUType = DeviceSKUType.A100
    num_devices_per_node: int = 8
    intra_node_bw_gb_per_s: float = 600.0  # NVLink 3.0 NVSwitch (8-GPU full mesh)

    @staticmethod
    def get_type():
        return NodeSKUType.A100_DGX


@dataclass
class H100DgxNodeSKUConfig(BaseNodeSKUConfig):
    device_sku_type: DeviceSKUType = DeviceSKUType.H100
    num_devices_per_node: int = 8
    intra_node_bw_gb_per_s: float = 900.0  # NVLink 4.0 NVSwitch (8-GPU full mesh)

    @staticmethod
    def get_type():
        return NodeSKUType.H100_DGX
