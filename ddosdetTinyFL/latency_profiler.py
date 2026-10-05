"""Inference Latency Profiling Utilities for TinyML Models.

This module provides comprehensive latency analysis for models after
applying quantization and pruning optimizations. It measures:
  - Per-Sample Inference Latency: Time to process a single sample (microseconds)
  - Batch Inference Latency: Time to process a batch (milliseconds)
  - Throughput: Number of samples processed per second
  - Percentile latencies: P50, P90, P95, P99 for robustness analysis
"""

import time
import torch
import statistics
from typing import Dict, List, Tuple, Any
from contextlib import contextmanager


def measure_inference_latency(
    model: torch.nn.Module,
    test_loader,
    warmup_batches: int = 2,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Measure inference latency per sample and batch on test data.
    
    This function performs inference on test data with warmup iterations to
    stabilize timing measurements. It records latency for each batch and
    sample to calculate statistics.
    
    Args:
        model: PyTorch model (typically after quantization and pruning)
        test_loader: DataLoader for test data
        warmup_batches: Number of batches to run before timing measurement
        device: Device to run inference on (default: CPU)
        
    Returns:
        Dictionary containing:
            - per_sample_latency_us: Latency per sample in microseconds
            - per_sample_latency_ms: Latency per sample in milliseconds
            - batch_latency_ms: Latency per batch in milliseconds
            - throughput_samples_per_sec: Number of samples processed per second
            - total_batches: Total number of batches processed
            - total_samples: Total number of samples processed
            - total_inference_time_sec: Total inference time in seconds
            - latency_percentiles: P50, P90, P95, P99 per-sample latencies
            - batch_latency_percentiles: P50, P90, P95, P99 batch latencies
            - inference_variability_percent: Coefficient of variation for latency
    """
    model.eval()
    device_obj = torch.device(device)
    
    # Warmup phase to stabilize timing
    with torch.no_grad():
        for batch_idx, (X_batch, _) in enumerate(test_loader):
            if batch_idx >= warmup_batches:
                break
            X_batch = X_batch.to(device_obj)
            _ = model(X_batch)
    
    # Measurement phase
    per_sample_latencies_us = []
    batch_latencies_ms = []
    total_batches = 0
    total_samples = 0
    total_inference_time = 0.0
    
    with torch.no_grad():
        for X_batch, _ in test_loader:
            batch_size = X_batch.size(0)
            X_batch = X_batch.to(device_obj)
            
            # Record batch latency
            start_time = time.perf_counter()
            _ = model(X_batch)
            end_time = time.perf_counter()
            
            batch_time_ms = (end_time - start_time) * 1000
            batch_latencies_ms.append(batch_time_ms)
            
            # Calculate per-sample latency
            per_sample_latency_ms = batch_time_ms / batch_size
            per_sample_latency_us = per_sample_latency_ms * 1000
            per_sample_latencies_us.extend([per_sample_latency_us] * batch_size)
            
            total_batches += 1
            total_samples += batch_size
            total_inference_time += (end_time - start_time)
    
    # Calculate statistics
    avg_per_sample_latency_us = statistics.mean(per_sample_latencies_us) if per_sample_latencies_us else 0.0
    avg_per_sample_latency_ms = avg_per_sample_latency_us / 1000
    avg_batch_latency_ms = statistics.mean(batch_latencies_ms) if batch_latencies_ms else 0.0
    
    # Throughput: samples per second
    throughput = total_samples / max(total_inference_time, 1e-9)
    
    # Percentile calculations for per-sample latency
    sorted_sample_latencies = sorted(per_sample_latencies_us)
    percentile_indices = {
        "p50": len(sorted_sample_latencies) // 2,
        "p90": int(len(sorted_sample_latencies) * 0.90),
        "p95": int(len(sorted_sample_latencies) * 0.95),
        "p99": int(len(sorted_sample_latencies) * 0.99),
    }
    
    sample_latency_percentiles = {
        key: sorted_sample_latencies[idx] if idx < len(sorted_sample_latencies) else 0.0
        for key, idx in percentile_indices.items()
    }
    
    # Percentile calculations for batch latency
    sorted_batch_latencies = sorted(batch_latencies_ms)
    batch_percentile_indices = {
        "p50": len(sorted_batch_latencies) // 2,
        "p90": int(len(sorted_batch_latencies) * 0.90),
        "p95": int(len(sorted_batch_latencies) * 0.95),
        "p99": int(len(sorted_batch_latencies) * 0.99),
    }
    
    batch_latency_percentiles = {
        key: sorted_batch_latencies[idx] if idx < len(sorted_batch_latencies) else 0.0
        for key, idx in batch_percentile_indices.items()
    }
    
    # Coefficient of Variation (CV) as a measure of variability
    std_dev = statistics.stdev(per_sample_latencies_us) if len(per_sample_latencies_us) > 1 else 0.0
    cv_percent = (std_dev / avg_per_sample_latency_us * 100) if avg_per_sample_latency_us > 0 else 0.0
    
    return {
        "per_sample_latency_us": float(avg_per_sample_latency_us),
        "per_sample_latency_ms": float(avg_per_sample_latency_ms),
        "batch_latency_ms": float(avg_batch_latency_ms),
        "throughput_samples_per_sec": float(throughput),
        "total_batches": int(total_batches),
        "total_samples": int(total_samples),
        "total_inference_time_sec": float(total_inference_time),
        "latency_percentiles": {
            "p50_us": float(sample_latency_percentiles["p50"]),
            "p90_us": float(sample_latency_percentiles["p90"]),
            "p95_us": float(sample_latency_percentiles["p95"]),
            "p99_us": float(sample_latency_percentiles["p99"]),
        },
        "batch_latency_percentiles": {
            "p50_ms": float(batch_latency_percentiles["p50"]),
            "p90_ms": float(batch_latency_percentiles["p90"]),
            "p95_ms": float(batch_latency_percentiles["p95"]),
            "p99_ms": float(batch_latency_percentiles["p99"]),
        },
        "inference_variability_percent": float(cv_percent),
    }


def analyze_model_latency(
    model: torch.nn.Module,
    test_loader,
    model_name: str = "model",
    batch_size: int = 32,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Comprehensive inference latency analysis for a model.
    
    This is the main function to call for complete latency profiling of
    quantized and pruned models.
    
    Args:
        model: PyTorch model (typically after quantization and pruning)
        test_loader: DataLoader for test data
        model_name: Name identifier for the model
        batch_size: Batch size used in inference
        device: Device to run inference on
        
    Returns:
        Dictionary containing:
            - model_name: Model identifier
            - batch_size: Batch size used
            - latency_metrics: Complete latency measurement results
            - interpretation: Human-readable summary of findings
    """
    latency_metrics = measure_inference_latency(model, test_loader, device=device)
    
    # Create interpretation
    interpretation = {
        "optimal_for_edge": latency_metrics["per_sample_latency_us"] < 50000,  # < 50ms
        "real_time_capable": latency_metrics["per_sample_latency_us"] < 100000,  # < 100ms
        "low_variability": latency_metrics["inference_variability_percent"] < 10,
        "min_latency_us": float(min([l for l in latency_metrics["latency_percentiles"].values()])),
        "max_latency_us": float(max([l for l in latency_metrics["latency_percentiles"].values()])),
    }
    
    return {
        "model_name": model_name,
        "batch_size": batch_size,
        "latency_metrics": latency_metrics,
        "interpretation": interpretation,
    }


def save_latency_metrics_to_file(
    metrics: Dict[str, Any],
    output_path: str = "latency_metrics.txt"
) -> None:
    """
    Save latency metrics to a human-readable text file.
    
    Args:
        metrics: Dictionary from analyze_model_latency()
        output_path: Path to save the metrics file
    """
    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("INFERENCE LATENCY ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Model: {metrics['model_name']}\n")
        f.write(f"Batch Size: {metrics['batch_size']}\n\n")
        
        # Main Latency Metrics
        f.write("MAIN LATENCY METRICS\n")
        f.write("-" * 70 + "\n")
        lat = metrics['latency_metrics']
        f.write(f"  Per-Sample Latency (Average):\n")
        f.write(f"    {lat['per_sample_latency_us']:.2f} μs (microseconds)\n")
        f.write(f"    {lat['per_sample_latency_ms']:.4f} ms (milliseconds)\n")
        f.write(f"  Batch Latency (Average): {lat['batch_latency_ms']:.2f} ms\n")
        f.write(f"  Throughput: {lat['throughput_samples_per_sec']:.2f} samples/sec\n\n")
        
        # Processing Statistics
        f.write("PROCESSING STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Total Batches Processed: {lat['total_batches']}\n")
        f.write(f"  Total Samples Processed: {lat['total_samples']}\n")
        f.write(f"  Total Inference Time: {lat['total_inference_time_sec']:.2f} seconds\n\n")
        
        # Percentile Analysis
        f.write("PERCENTILE LATENCY ANALYSIS (Per-Sample)\n")
        f.write("-" * 70 + "\n")
        percentiles = lat['latency_percentiles']
        f.write(f"  P50 (Median):  {percentiles['p50_us']:.2f} μs\n")
        f.write(f"  P90:           {percentiles['p90_us']:.2f} μs\n")
        f.write(f"  P95:           {percentiles['p95_us']:.2f} μs\n")
        f.write(f"  P99:           {percentiles['p99_us']:.2f} μs\n\n")
        
        # Batch Percentile Analysis
        f.write("PERCENTILE LATENCY ANALYSIS (Per-Batch)\n")
        f.write("-" * 70 + "\n")
        batch_percentiles = lat['batch_latency_percentiles']
        f.write(f"  P50 (Median):  {batch_percentiles['p50_ms']:.2f} ms\n")
        f.write(f"  P90:           {batch_percentiles['p90_ms']:.2f} ms\n")
        f.write(f"  P95:           {batch_percentiles['p95_ms']:.2f} ms\n")
        f.write(f"  P99:           {batch_percentiles['p99_ms']:.2f} ms\n\n")
        
        # Variability and Consistency
        f.write("CONSISTENCY & VARIABILITY\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Coefficient of Variation: {lat['inference_variability_percent']:.2f}%\n")
        interp = metrics['interpretation']
        f.write(f"  Optimal for Edge Devices: {'Yes' if interp['optimal_for_edge'] else 'No'}\n")
        f.write(f"  Real-Time Capable: {'Yes' if interp['real_time_capable'] else 'No'}\n")
        f.write(f"  Low Variability: {'Yes' if interp['low_variability'] else 'No'}\n\n")
        
        # Range Information
        f.write("LATENCY RANGE\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Minimum Latency: {interp['min_latency_us']:.2f} μs\n")
        f.write(f"  Maximum Latency: {interp['max_latency_us']:.2f} μs\n")
        f.write(f"  Latency Range: {interp['max_latency_us'] - interp['min_latency_us']:.2f} μs\n")
        f.write("=" * 70 + "\n")
