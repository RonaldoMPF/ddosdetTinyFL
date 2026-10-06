"""Experiment runner for standard and optimized FL scenarios.

This module keeps the experimental variables centralized while reusing the metrics
already implemented in the repository. It exports the metrics from both scenarios
in JSON/CSV so they can be plotted and compared with the same data model.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

from flwr.app import ArrayRecord, ConfigRecord, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from ddosdetTinyFL.client_app import weighted_average_eval, weighted_average_train
from ddosdetTinyFL.experimental_config import ExperimentalConfig
from ddosdetTinyFL.latency_profiler import profile_model_runtime_and_energy
from ddosdetTinyFL.memory_profiler import analyze_model_memory_footprint
from ddosdetTinyFL.metrics_collector import ComparativeMetricsAnalyzer, MetricsCollector
from ddosdetTinyFL.task import DDoSClassifier, evaluator, load_centralized_dataset


def apply_model_optimization(
    model: nn.Module,
    prune_ratio: float = 0.25,
    enable_pruning: bool = True,
    enable_quantization: bool = True,
) -> nn.Module:
    """Apply pruning and/or quantization to a model in a deterministic way."""
    if enable_pruning:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=prune_ratio)
                prune.remove(module, "weight")

    if enable_quantization:
        quantize_(model, Int8DynamicActivationInt8WeightConfig())

    return model


def standard_global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate the FP32 global model on the centralized test set."""
    model = DDoSClassifier()
    model.load_state_dict(arrays.to_torch_state_dict())
    test_loader = load_centralized_dataset()
    loss, accuracy, extra_metrics = evaluator(model, test_loader)
    return MetricRecord(
        {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "f1_score": float(extra_metrics["f1_score"]),
            "precision": float(extra_metrics["precision"]),
            "recall": float(extra_metrics["recall"]),
        }
    )


def optimized_global_evaluate(server_round: int, arrays: ArrayRecord, prune_ratio: float = 0.25) -> MetricRecord:
    """Evaluate the pruned + quantized global model on the centralized test set."""
    model = DDoSClassifier()
    model.load_state_dict(arrays.to_torch_state_dict())
    apply_model_optimization(model, prune_ratio=prune_ratio, enable_pruning=True, enable_quantization=True)
    test_loader = load_centralized_dataset()
    loss, accuracy, extra_metrics = evaluator(model, test_loader)
    return MetricRecord(
        {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "f1_score": float(extra_metrics["f1_score"]),
            "precision": float(extra_metrics["precision"]),
            "recall": float(extra_metrics["recall"]),
        }
    )


def run_single_scenario(
    scenario_name: str,
    grid: Grid,
    config: ExperimentalConfig,
    optimize_model: bool = False,
) -> Dict[str, Any]:
    """Run one Flower scenario (standard or optimized) with FedAvg and export the metrics."""
    from flwr.serverapp import ServerApp

    app = ServerApp()

    if optimize_model:
        evaluate_fn = lambda rnd, arrays: optimized_global_evaluate(rnd, arrays, prune_ratio=config.prune_ratio)
    else:
        evaluate_fn = standard_global_evaluate

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=config.fraction_evaluate,
        min_fit_clients=1,
        min_evaluate_clients=1,
        evaluate_metrics_aggregation_fn=weighted_average_eval,
        fit_metrics_aggregation_fn=weighted_average_train,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(DDoSClassifier().state_dict()),
        train_config=ConfigRecord({"lr": config.learning_rate}),
        evaluate_config=ConfigRecord({"lr": config.learning_rate}),
        num_rounds=config.num_server_rounds,
        evaluate_fn=evaluate_fn,
    )

    final_state_dict = result.arrays.to_torch_state_dict()
    final_model = DDoSClassifier()
    final_model.load_state_dict(final_state_dict)

    if optimize_model:
        final_model = apply_model_optimization(
            final_model,
            prune_ratio=config.prune_ratio,
            enable_pruning=config.enable_pruning,
            enable_quantization=config.enable_quantization,
        )

    test_loader = load_centralized_dataset()
    loss, accuracy, extra_metrics = evaluator(final_model, test_loader)
    final_metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "f1_score": float(extra_metrics["f1_score"]),
        "precision": float(extra_metrics["precision"]),
        "recall": float(extra_metrics["recall"]),
    }

    latency_profile = profile_model_runtime_and_energy(
        final_model,
        test_loader,
        model_name=scenario_name,
        batch_size=config.batch_size,
        prune_ratio=config.prune_ratio,
        quantized=optimize_model and config.enable_quantization,
    )

    memory_profile = analyze_model_memory_footprint(
        final_model,
        test_loader,
        model_name=scenario_name,
    )

    collector = MetricsCollector(scenario_name=scenario_name)
    collector.record_global_metrics({**final_metrics, **{
        "latency_per_sample_us": latency_profile["latency_metrics"]["per_sample_latency_us"],
        "batch_latency_ms": latency_profile["latency_metrics"]["batch_latency_ms"],
        "throughput_samples_per_sec": latency_profile["latency_metrics"]["throughput_samples_per_sec"],
        "flash_memory_mb": memory_profile["summary"]["flash_memory_mb"],
        "sram_peak_memory_mb": memory_profile["summary"]["sram_peak_memory_mb"],
        "compression_ratio": memory_profile["summary"]["compression_ratio"],
    }})

    collector.record_round_metrics(0, final_metrics, client_count=config.num_partitions)
    output_dir = Path(config.metric_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    standard_path = output_dir / config.standard_metrics_json if scenario_name == "standard" else output_dir / config.optimized_metrics_json
    collector.export_to_json(str(standard_path))
    collector.export_to_csv(str(output_dir / f"{scenario_name}_metrics.csv"))

    return {
        "scenario": scenario_name,
        "final_metrics": final_metrics,
        "latency_profile": latency_profile,
        "memory_profile": memory_profile,
        "export_path": str(standard_path),
    }


def compare_scenarios(config: ExperimentalConfig, output_dir: str = None) -> Dict[str, Any]:
    """Load standard and optimized exports and produce comparison CSV/JSON for plotting."""
    output_dir = Path(output_dir or config.metric_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    std_json = output_dir / config.standard_metrics_json
    opt_json = output_dir / config.optimized_metrics_json

    if not std_json.exists() or not opt_json.exists():
        raise FileNotFoundError("Standard and optimized scenario exports are required before comparison.")

    with open(std_json, "r", encoding="utf-8") as f:
        std_data = json.load(f)
    with open(opt_json, "r", encoding="utf-8") as f:
        opt_data = json.load(f)

    analyzer = ComparativeMetricsAnalyzer(
        MetricsCollector(scenario_name="standard"),
        MetricsCollector(scenario_name="optimized"),
    )
    analyzer.standard.round_metrics = std_data.get("round_metrics", {})
    analyzer.standard.global_metrics = std_data.get("global_metrics", {})
    analyzer.optimized.round_metrics = opt_data.get("round_metrics", {})
    analyzer.optimized.global_metrics = opt_data.get("global_metrics", {})

    comparison_csv = output_dir / config.comparison_metrics_csv
    analyzer.export_comparison_to_csv(str(comparison_csv))

    comparison_json = output_dir / config.comparison_metrics_json
    analyzer.export_comparison_to_json(str(comparison_json))

    return {
        "comparison_csv": str(comparison_csv),
        "comparison_json": str(comparison_json),
        "delta_metrics": analyzer.compute_delta_metrics(),
    }


def run_experiment(config: ExperimentalConfig, grid: Grid) -> Dict[str, Any]:
    """Execute standard and optimized scenarios and export the comparison data."""
    results = {}
    if config.run_standard_scenario:
        results["standard"] = run_single_scenario("standard", grid, config, optimize_model=False)
    if config.run_optimized_scenario:
        results["optimized"] = run_single_scenario("optimized", grid, config, optimize_model=True)

    if config.run_standard_scenario and config.run_optimized_scenario:
        comparison = compare_scenarios(config, output_dir=config.metric_output_dir)
        results["comparison"] = comparison

    return results
