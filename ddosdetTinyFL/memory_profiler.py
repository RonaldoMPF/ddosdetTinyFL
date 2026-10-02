"""Memory Profiling Utilities for Flash Memory and SRAM Peak Memory Metrics.

This module provides comprehensive memory footprint analysis for models after
applying quantization and pruning optimizations. It measures:
  - Flash Memory Footprint: Static storage usage in Flash memory (bytes)
  - SRAM Peak Memory: Peak dynamic RAM consumption during inference (bytes)
"""

import os
import io
import torch
import psutil
import tracemalloc
from typing import Dict, Tuple, Any
from contextlib import contextmanager


def calculate_flash_memory_footprint(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Calculate Flash Memory Footprint (static storage usage).
    
    Measures the size of the model when saved to disk, representing the storage
    space required on Flash memory for embedding in edge/IoT devices.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Dictionary containing:
            - flash_memory_bytes: Model size in bytes
            - flash_memory_kb: Model size in kilobytes
            - flash_memory_mb: Model size in megabytes
            - model_params: Total number of parameters
            - model_params_million: Total parameters in millions
    """
    # Save model to bytes buffer
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    
    # Get the size of the serialized model
    flash_memory_bytes = buffer.getbuffer().nbytes
    flash_memory_kb = flash_memory_bytes / 1024
    flash_memory_mb = flash_memory_kb / 1024
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_params_million = total_params / 1e6
    
    return {
        "flash_memory_bytes": int(flash_memory_bytes),
        "flash_memory_kb": float(flash_memory_kb),
        "flash_memory_mb": float(flash_memory_mb),
        "model_params": int(total_params),
        "model_params_million": float(total_params_million),
    }


@contextmanager
def measure_sram_peak_memory(model: torch.nn.Module):
    """
    Context manager to measure peak SRAM memory consumption during inference.
    
    Traces memory allocation during model inference to capture peak RAM usage,
    which is critical for embedded devices with limited memory.
    
    Yields:
        A dictionary to store measurement results
    """
    measurement = {"peak_memory_bytes": 0, "peak_memory_mb": 0.0}
    
    # Start tracing memory allocations
    tracemalloc.start()
    
    try:
        yield measurement
    finally:
        # Get the peak memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Convert to appropriate units
        measurement["peak_memory_bytes"] = int(peak)
        measurement["peak_memory_mb"] = float(peak / (1024 ** 2))


def measure_inference_sram_peak(
    model: torch.nn.Module,
    test_loader,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Measure peak SRAM memory consumption during model inference on a test dataset.
    
    This function runs inference on the provided test data and records the peak
    memory usage, which represents the maximum RAM needed during operation.
    
    Args:
        model: PyTorch model to profile
        test_loader: DataLoader for test data
        device: Device to run inference on (default: CPU)
        
    Returns:
        Dictionary containing:
            - sram_peak_memory_bytes: Peak RAM used during inference in bytes
            - sram_peak_memory_mb: Peak RAM used during inference in MB
            - inference_batch_count: Number of batches processed
            - avg_batch_memory_mb: Average memory per batch in MB
    """
    model.eval()
    device_obj = torch.device(device)
    
    with measure_sram_peak_memory(model) as measurement:
        with torch.no_grad():
            batch_count = 0
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device_obj)
                _ = model(X_batch)
                batch_count += 1
    
    avg_batch_memory = measurement["peak_memory_mb"] / max(batch_count, 1)
    
    return {
        "sram_peak_memory_bytes": measurement["peak_memory_bytes"],
        "sram_peak_memory_mb": measurement["peak_memory_mb"],
        "inference_batch_count": batch_count,
        "avg_batch_memory_mb": float(avg_batch_memory),
    }


def get_system_memory_info() -> Dict[str, Any]:
    """
    Get current system memory information.
    
    Returns:
        Dictionary containing:
            - total_ram_mb: Total system RAM in MB
            - available_ram_mb: Available RAM in MB
            - used_ram_mb: Used RAM in MB
            - memory_utilization_percent: Current memory utilization percentage
    """
    memory_info = psutil.virtual_memory()
    
    return {
        "total_ram_mb": float(memory_info.total / (1024 ** 2)),
        "available_ram_mb": float(memory_info.available / (1024 ** 2)),
        "used_ram_mb": float(memory_info.used / (1024 ** 2)),
        "memory_utilization_percent": float(memory_info.percent),
    }


def analyze_model_memory_footprint(
    model: torch.nn.Module,
    test_loader,
    model_name: str = "model",
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Comprehensive memory footprint analysis combining Flash and SRAM metrics.
    
    This is the main function to call for complete memory profiling of quantized
    and pruned models. It provides Flash memory footprint and peak SRAM consumption.
    
    Args:
        model: PyTorch model (typically after quantization and pruning)
        test_loader: DataLoader for test data
        model_name: Name identifier for the model
        device: Device to run inference on
        
    Returns:
        Dictionary containing:
            - model_name: Model identifier
            - flash_memory: Flash memory footprint metrics
            - sram_memory: Peak SRAM memory metrics
            - system_memory: Current system memory status
            - total_optimized_size_mb: Total size combining Flash and SRAM info
    """
    # Get system info before profiling
    system_info_before = get_system_memory_info()
    
    # Calculate Flash memory footprint
    flash_metrics = calculate_flash_memory_footprint(model)
    
    # Measure SRAM peak during inference
    sram_metrics = measure_inference_sram_peak(model, test_loader, device)
    
    # Get system info after profiling
    system_info_after = get_system_memory_info()
    
    return {
        "model_name": model_name,
        "flash_memory": flash_metrics,
        "sram_memory": sram_metrics,
        "system_memory_before": system_info_before,
        "system_memory_after": system_info_after,
        "summary": {
            "flash_memory_mb": flash_metrics["flash_memory_mb"],
            "sram_peak_memory_mb": sram_metrics["sram_peak_memory_mb"],
            "total_model_params_million": flash_metrics["model_params_million"],
            "compression_ratio": _calculate_compression_ratio(flash_metrics),
        }
    }


def _calculate_compression_ratio(flash_metrics: Dict[str, Any]) -> float:
    """
    Calculate compression ratio based on parameters vs. flash size.
    
    Assumes FP32 (4 bytes per parameter) as baseline.
    
    Args:
        flash_metrics: Flash memory metrics dictionary
        
    Returns:
        Compression ratio (baseline_size / actual_size)
    """
    baseline_size_bytes = flash_metrics["model_params"] * 4  # FP32 = 4 bytes
    actual_size_bytes = flash_metrics["flash_memory_bytes"]
    
    if actual_size_bytes == 0:
        return 1.0
    
    return float(baseline_size_bytes / actual_size_bytes)


def save_memory_metrics_to_file(
    metrics: Dict[str, Any],
    output_path: str = "memory_metrics.txt"
) -> None:
    """
    Save memory metrics to a human-readable text file.
    
    Args:
        metrics: Dictionary from analyze_model_memory_footprint()
        output_path: Path to save the metrics file
    """
    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("MEMORY FOOTPRINT ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Model: {metrics['model_name']}\n\n")
        
        # Flash Memory Section
        f.write("FLASH MEMORY FOOTPRINT (Static Storage Usage)\n")
        f.write("-" * 70 + "\n")
        flash = metrics['flash_memory']
        f.write(f"  Flash Memory Size: {flash['flash_memory_mb']:.2f} MB\n")
        f.write(f"                     {flash['flash_memory_kb']:.2f} KB\n")
        f.write(f"                     {flash['flash_memory_bytes']} bytes\n")
        f.write(f"  Total Parameters: {flash['model_params']:,} params\n")
        f.write(f"                    {flash['model_params_million']:.2f}M params\n\n")
        
        # SRAM Memory Section
        f.write("SRAM PEAK MEMORY (Dynamic RAM Consumption)\n")
        f.write("-" * 70 + "\n")
        sram = metrics['sram_memory']
        f.write(f"  Peak SRAM Usage: {sram['sram_peak_memory_mb']:.2f} MB\n")
        f.write(f"                   {sram['sram_peak_memory_bytes']} bytes\n")
        f.write(f"  Batches Processed: {sram['inference_batch_count']}\n")
        f.write(f"  Avg Memory/Batch: {sram['avg_batch_memory_mb']:.2f} MB\n\n")
        
        # System Memory Section
        f.write("SYSTEM MEMORY STATUS (Before Profiling)\n")
        f.write("-" * 70 + "\n")
        sys_before = metrics['system_memory_before']
        f.write(f"  Total System RAM: {sys_before['total_ram_mb']:.2f} MB\n")
        f.write(f"  Available RAM: {sys_before['available_ram_mb']:.2f} MB\n")
        f.write(f"  Used RAM: {sys_before['used_ram_mb']:.2f} MB\n")
        f.write(f"  Utilization: {sys_before['memory_utilization_percent']:.1f}%\n\n")
        
        # Summary Section
        f.write("SUMMARY\n")
        f.write("-" * 70 + "\n")
        summary = metrics['summary']
        f.write(f"  Flash Memory Footprint: {summary['flash_memory_mb']:.2f} MB\n")
        f.write(f"  SRAM Peak Memory: {summary['sram_peak_memory_mb']:.2f} MB\n")
        f.write(f"  Total Model Parameters: {summary['total_model_params_million']:.2f}M\n")
        f.write(f"  Compression Ratio: {summary['compression_ratio']:.2f}x\n")
        f.write("=" * 70 + "\n")
