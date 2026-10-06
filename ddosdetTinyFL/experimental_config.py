"""Experimental Configuration Management for DDoS Detection FL.

This module centralizes all experimental variables, making it easy to adjust
parameters for both standard and optimized (quantized + pruned) model scenarios.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import json


@dataclass
class ExperimentalConfig:
    """Configuration for federated learning experiments with standard and optimized models."""
    
    # ============ FEDERATED LEARNING CONFIGURATION ============
    num_server_rounds: int = 10
    num_partitions: int = 6
    fraction_evaluate: float = 1.0
    
    # ============ TRAINING CONFIGURATION ============
    batch_size: int = 32
    learning_rate: float = 0.001
    local_epochs: int = 1
    
    # ============ MODEL OPTIMIZATION CONFIGURATION ============
    # Pruning parameters
    enable_pruning: bool = True
    pruning_amount: float = 0.25  # Prune 25% of weights (L1 unstructured)
    prune_at_client: bool = True  # Apply pruning at client-side
    prune_at_server: bool = False  # Apply pruning at server-side for centralized eval
    
    # Quantization parameters
    enable_quantization: bool = True
    quantization_dtype: str = "int8"  # int8 for Dynamic INT8 quantization
    quantize_at_eval: bool = True  # Apply quantization only during evaluation
    
    # ============ SCENARIO FLAGS ============
    # These control which scenarios to run during the experiment
    run_standard_scenario: bool = True  # Run unoptimized (FP32) model scenario
    run_optimized_scenario: bool = True  # Run optimized (pruned + quantized) model scenario
    
    # ============ METRICS & PROFILING ============
    profile_latency: bool = True  # Profile inference latency
    profile_memory: bool = True  # Profile memory footprint
    profile_energy: bool = True  # Profile estimated energy consumption
    
    # ============ DATA CONFIGURATION ============
    # Non-IID partitioning with Dirichlet distribution
    dirichlet_alpha: float = 1.0  # Lower alpha = more non-IID
    train_test_split_ratio: float = 0.7  # 70% train, 30% validation per client
    random_seed: int = 42
    
    # ============ EDGE DEVICE SIMULATION PARAMETERS ============
    node_voltage_v: float = 3.3  # Simulated node voltage in volts
    node_frequency_mhz: float = 400.0  # Simulated node frequency in MHz
    
    # ============ OUTPUT PATHS ============
    metrics_output_dir: str = "experiment_results"
    standard_metrics_file: str = "standard_model_metrics.json"
    optimized_metrics_file: str = "optimized_model_metrics.json"
    comparison_metrics_file: str = "comparison_metrics.json"
    plots_output_dir: str = "plots"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @staticmethod
    def from_json(path: str) -> 'ExperimentalConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return ExperimentalConfig(**data)


@dataclass
class ScenarioMetrics:
    """Container for metrics collected during a scenario (standard or optimized)."""
    
    scenario_name: str  # "standard" or "optimized"
    fl_rounds_metrics: Dict[int, Dict[str, Any]] = None  # Per-round metrics
    final_global_metrics: Dict[str, Any] = None  # Final evaluation metrics
    latency_profile: Dict[str, Any] = None  # Latency profiling results
    memory_profile: Dict[str, Any] = None  # Memory profiling results
    training_summary: Dict[str, Any] = None  # Aggregated training metrics
    
    def __post_init__(self):
        """Initialize nested dictionaries if None."""
        if self.fl_rounds_metrics is None:
            self.fl_rounds_metrics = {}
        if self.final_global_metrics is None:
            self.final_global_metrics = {}
        if self.training_summary is None:
            self.training_summary = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'scenario_name': self.scenario_name,
            'fl_rounds_metrics': self.fl_rounds_metrics,
            'final_global_metrics': self.final_global_metrics,
            'latency_profile': self.latency_profile,
            'memory_profile': self.memory_profile,
            'training_summary': self.training_summary,
        }
    
    def to_json(self, path: str) -> None:
        """Save metrics to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @staticmethod
    def from_json(path: str) -> 'ScenarioMetrics':
        """Load metrics from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return ScenarioMetrics(**data)


def get_default_config() -> ExperimentalConfig:
    """Get default experimental configuration."""
    return ExperimentalConfig()


def create_custom_config(**kwargs) -> ExperimentalConfig:
    """Create a custom experimental configuration by overriding defaults.
    
    Example:
        config = create_custom_config(
            num_server_rounds=20,
            batch_size=64,
            enable_pruning=False
        )
    """
    default = get_default_config()
    for key, value in kwargs.items():
        if hasattr(default, key):
            setattr(default, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    return default
