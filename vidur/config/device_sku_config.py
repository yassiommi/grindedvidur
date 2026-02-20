from dataclasses import dataclass, field

from vidur.config.base_fixed_config import BaseFixedConfig
from vidur.logger import init_logger
from vidur.types import DeviceSKUType

logger = init_logger(__name__)


@dataclass
class BaseDeviceSKUConfig(BaseFixedConfig):
    fp16_tflops: int
    total_memory_gb: int
    # HBM bandwidth in GB/s (raw peak, before efficiency factor)
    memory_bandwidth_gb_per_s: float = 0.0
    # PCIe bandwidth in GB/s (unidirectional, for host<->device transfers)
    pcie_bandwidth_gb_per_s: float = 0.0


@dataclass
class A40DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 150
    total_memory_gb: int = 45
    memory_bandwidth_gb_per_s: float = 696.0  # GDDR6X
    pcie_bandwidth_gb_per_s: float = 31.5  # PCIe Gen4 x16

    @staticmethod
    def get_type():
        return DeviceSKUType.A40


@dataclass
class A100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 312
    total_memory_gb: int = 80
    memory_bandwidth_gb_per_s: float = 2039.0  # HBM2e
    pcie_bandwidth_gb_per_s: float = 31.5  # PCIe Gen4 x16

    @staticmethod
    def get_type():
        return DeviceSKUType.A100


@dataclass
class H100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 1000
    total_memory_gb: int = 80
    memory_bandwidth_gb_per_s: float = 3350.0  # HBM3
    pcie_bandwidth_gb_per_s: float = 64.0  # PCIe Gen5 x16

    @staticmethod
    def get_type():
        return DeviceSKUType.H100
