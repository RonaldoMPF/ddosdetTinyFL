"""Inference Latency Profiling Utilities for TinyML Models.

This module provides comprehensive latency analysis for models after
applying quantization and pruning optimizations. It measures:
  - Per-Sample Inference Latency: Time to process a single sample (microseconds)
  - Batch Inference Latency: Time to process a batch (milliseconds)
  - Throughput: Number of samples processed per second
  - Percentile latencies: P50, P90, P95, P99 for robustness analysis
  - Estimated electrical consumption: Energy associated with the execution cycle
    on a Flower simulation node using a lightweight hardware proxy model.
"""

import time
import statistics
from typing import Dict, Any

import torch


DEFAULT_NODE_VOLTAGE_V = 3.3
DEFAULT_NODE_FREQUENCY_MHZ = 400.0
DEFAULT_IDLE_POWER_W = 0.08


def _estimate_active_power_w(
    model: torch.nn.Module,
    batch_size: int,
    prune_ratio: float = 0.25,
    quantized: bool = True,
    voltage_v: float = DEFAULT_NODE_VOLTAGE_V,
    frequency_mhz: float = DEFAULT_NODE_FREQUENCY_MHZ,
) -> float:
    """Estimate the effective average power of a model execution on a node.

    We use a lightweight proxy model rather than direct hardware metering, which
    matches the Flower simulation setting where there is no physical device power
    sensor. The estimate scales with parameter count, pruning, quantization,
    and nominal node voltage/frequency.
    """
    total_params = sum(p.numel() for p in model.parameters())
    if total_params <= 0:
        return DEFAULT_IDLE_POWER_W

    # Approximate multiply-accumulate workload and the reduction caused by
    # pruning and lower-bit quantization. A pruned model executes fewer useful
    # operations, and INT8 quantization reduces the bit-width and energy per op.
    workload_factor = 2.0 * total_params / max(batch_size, 1)
    pruning_factor = 1.0 - min(max(prune_ratio, 0.0), 0.9) * 0.7
    quantization_factor = 0.25 if quantized else 1.0
    frequency_factor = (frequency_mhz / 400.0)
    voltage_factor = voltage_v / 3.3

    # Base power intentionally stays small and is dominated by the workload and
    # the node's nominal electrical conditions.
    estimate = (
        DEFAULT_IDLE_POWER_W
        + 2.5e-7 * workload_factor * pruning_factor * quantization_factor
        * frequency_factor * voltage_factor
    )
    return float(max(estimate, DEFAULT_IDLE_POWER_W))


def estimate_execution_cycle_energy(
    model: torch.nn.Module,
    latency_ms: float,
    batch_size: int = 32,
    prune_ratio: float = 0.25,
    quantized: bool = True,
    voltage_v: float = DEFAULT_NODE_VOLTAGE_V,
    frequency_mhz: float = DEFAULT_NODE_FREQUENCY_MHZ,
) -> Dict[str, float]:
    """Estimate the electrical energy associated with one inference cycle.

    The metric is intentionally implemented as a host-side proxy for a Flower
    simulation node, where no physical power monitor exists. The estimate is
    based on average execution power and measured inference latency.

    Returns:
        {
            "estimated_average_power_w": ...,
            "energy_per_execution_cycle_j": ...,
            "energy_per_sample_mj": ...,
            "energy_per_batch_j": ...,
        }
    """
    latency_sec = max(latency_ms / 1000.0, 1e-9)
    average_power_w = _estimate_active_power_w(
        model=model,
        batch_size=batch_size,
        prune_ratio=prune_ratio,
        quantized=quantized,
        voltage_v=voltage_v,
        frequency_mhz=frequency_mhz,
    )
    energy_per_execution_cycle_j = average_power_w * latency_sec
    energy_per_sample_mj = (energy_per_execution_cycle_j / max(batch_size, 1)) * 1000.0
    energy_per_batch_j = energy_per_execution_cycle_j

    return {
        "estimated_average_power_w": float(average_power_w),
        "energy_per_execution_cycle_j": float(energy_per_execution_cycle_j),
        "energy_per_sample_mj": float(energy_per_sample_mj),
        "energy_per_batch_j": float(energy_per_batch_j),
    }


def measure_inference_latency(
    model: torch.nn.Module,
    test_loader,
    warmup_batches: int = 2,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Measure inference latency per sample and batch on test data.
    
    This function performs inference on test data with warmup iterations to
    stabilize timing measurements. It records latency for each batch and
    sample to calculate statistics.
    """
    model.eval()
    device_obj = torch.device(device)

    with torch.no_grad():
        for batch_idx, (X_batch, _) in enumerate(test_loader):
            if batch_idx >= warmup_batches:
                break
            X_batch = X_batch.to(device_obj)
            _ = model(X_batch)

    per_sample_latencies_us = []
    batch_latencies_ms = []
    total_batches = 0
    total_samples = 0
    total_inference_time = 0.0

    with torch.no_grad():
        for X_batch, _ in test_loader:
            batch_size = X_batch.size(0)
            X_batch = X_batch.to(device_obj)

            start_time = time.perf_counter()
            _ = model(X_batch)
            end_time = time.perf_counter()

            batch_time_ms = (end_time - start_time) * 1000.0
            batch_latencies_ms.append(batch_time_ms)

            per_sample_latency_ms = batch_time_ms / batch_size
            per_sample_latency_us_value = per_sample_latency_ms * 1000.0
            per_sample_latencies_us.extend([per_sample_latency_us_value] * batch_size)

            total_batches += 1
            total_samples += batch_size
            total_inference_time += (end_time - start_time)

    avg_per_sample_latency_us = statistics.mean(per_sample_latencies_us) if per_sample_latencies_us else 0.0
    avg_per_sample_latency_ms = avg_per_sample_latency_us / 1000.0
    avg_batch_latency_ms = statistics.mean(batch_latencies_ms) if batch_latencies_ms else 0.0
    throughput = total_samples / max(total_inference_time, 1e-9)

    sorted_sample_latencies = sorted(per_sample_latencies_us)
    sorted_batch_latencies = sorted(batch_latencies_ms)

    def percentile(values, pct):
        if not values:
            return 0.0
        index = min(len(values) - 1, max(0, int(len(values) * pct)))
        return float(values[index])

    return {
        "per_sample_latency_us": float(avg_per_sample_latency_us),
        "per_sample_latency_ms": float(avg_per_sample_latency_ms),
        "batch_latency_ms": float(avg_batch_latency_ms),
        "throughput_samples_per_sec": float(throughput),
        "total_batches": int(total_batches),
        "total_samples": int(total_samples),
        "total_inference_time_sec": float(total_inference_time),
        "latency_percentiles": {
            "p50_us": float(percentile(sorted_sample_latencies, 0.50)),
            "p90_us": float(percentile(sorted_sample_latencies, 0.90)),
            "p95_us": float(percentile(sorted_sample_latencies, 0.95)),
            "p99_us": float(percentile(sorted_sample_latencies, 0.99)),
        },
        "batch_latency_percentiles": {
            "p50_ms": float(percentile(sorted_batch_latencies, 0.50)),
            "p90_ms": float(percentile(sorted_batch_latencies, 0.90)),
            "p95_ms": float(percentile(sorted_batch_latencies, 0.95)),
            "p99_ms": float(percentile(sorted_batch_latencies, 0.99)),
        },
        "inference_variability_percent": float(
            (statistics.stdev(per_sample_latencies_us) / avg_per_sample_latency_us * 100.0)
            if len(per_sample_latencies_us) > 1 and avg_per_sample_latency_us > 0 else 0.0
        ),
    }


def profile_model_runtime_and_energy(
    model: torch.nn.Module,
    test_loader,
    model_name: str = "model",
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
    prune_ratio: float = 0.25,
    quantized: bool = True,
    node_voltage_v: float = DEFAULT_NODE_VOLTAGE_V,
    node_frequency_mhz: float = DEFAULT_NODE_FREQUENCY_MHZ,
) -> Dict[str, Any]:
    """Collect latency and estimated electrical-energy metrics for a model."""
    latency_metrics = measure_inference_latency(model, test_loader, device=device)
    energy_metrics = estimate_execution_cycle_energy(
        model=model,
        latency_ms=latency_metrics["batch_latency_ms"],
        batch_size=batch_size,
        prune_ratio=prune_ratio,
        quantized=quantized,
        voltage_v=node_voltage_v,
        frequency_mhz=node_frequency_mhz,
    )

    interpretation = {
        "optimal_for_edge": latency_metrics["per_sample_latency_us"] < 50000.0,
        "real_time_capable": latency_metrics["per_sample_latency_us"] < 100000.0,
        "low_variability": latency_metrics["inference_variability_percent"] < 10.0,
        "min_latency_us": float(min(latency_metrics["latency_percentiles"].values())),
        "max_latency_us": float(max(latency_metrics["latency_percentiles"].values())),
    }

    return {
        "model_name": model_name,
        "batch_size": batch_size,
        "latency_metrics": latency_metrics,
        "energy_metrics": energy_metrics,
        "interpretation": interpretation,
    }


def save_latency_metrics_to_file(
    metrics: Dict[str, Any],
    output_path: str = "latency_metrics.txt",
) -> None:
    """Save latency metrics to a human-readable text file."""
    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("INFERENCE LATENCY AND ENERGY ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Model: {metrics['model_name']}\n")
        f.write(f"Batch Size: {metrics['batch_size']}\n\n")

        lat = metrics['latency_metrics']
        f.write("MAIN LATENCY METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Per-Sample Latency (Average): {lat['per_sample_latency_us']:.2f} μs\n")
        f.write(f"  Per-Sample Latency (Average): {lat['per_sample_latency_ms']:.4f} ms\n")
        f.write(f"  Batch Latency (Average): {lat['batch_latency_ms']:.2f} ms\n")
        f.write(f"  Throughput: {lat['throughput_samples_per_sec']:.2f} samples/sec\n\n")

        eng = metrics['energy_metrics']
        f.write("ESTIMATED ELECTRICAL ENERGY\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Estimated Average Power: {eng['estimated_average_power_w']:.6f} W\n")
        f.write(f"  Energy per Execution Cycle: {eng['energy_per_execution_cycle_j']:.6f} J\n")
        f.write(f"  Energy per Sample: {eng['energy_per_sample_mj']:.3f} mJ\n")
        f.write(f"  Energy per Batch: {eng['energy_per_batch_j']:.6f} J\n\n")

        f.write("PERCENTILE LATENCY ANALYSIS (Per-Sample)\n")
        f.write("-" * 70 + "\n")
        for name, value in lat['latency_percentiles'].items():
            f.write(f"  {name}: {value:.2f} μs\n")
        f.write("\n")

        f.write("CONSISTENCY & VARIABILITY\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Coefficient of Variation: {lat['inference_variability_percent']:.2f}%\n")
        f.write(f"  Real-Time Capable: {'Yes' if metrics['interpretation']['real_time_capable'] else 'No'}\n")
        f.write("=" * 70 + "\n")
