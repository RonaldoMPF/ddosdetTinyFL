"""Network Metrics Collection and Analysis for Federated Learning.

This module provides comprehensive network performance metrics collection,
following the same collection standard as other metrics (latency_profiler.py
and memory_profiler.py). It measures:
  - Communication Bandwidth: Bytes transmitted per round
  - Compression Efficiency: Reduction in message size
  - Transmission Time: Estimated time to send model updates
  - Cumulative Network Cost: Total bytes over all rounds
  - Gradient Quantization Impact: Effects of quantization on model quality
"""

from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
from ddosdetTinyFL.compression import (
    GradientQuantizer,
    WeightCompressor,
    DeltaCompressor,
    BandwidthEstimator,
)


class NetworkMetricsCollector:
    """Collects and aggregates network metrics for federated learning rounds."""
    
    def __init__(
        self,
        network_bandwidth_mbps: float = 100.0,
        network_latency_ms: float = 10.0,
        gradient_bits: int = 8,
    ):
        """
        Initialize network metrics collector.
        
        Args:
            network_bandwidth_mbps: Available network bandwidth in Mbps
            network_latency_ms: Network latency in milliseconds
            gradient_bits: Bits for gradient quantization
        """
        self.network_bandwidth_mbps = network_bandwidth_mbps
        self.network_latency_ms = network_latency_ms
        self.gradient_bits = gradient_bits
        
        self.gradient_quantizer = GradientQuantizer(bits=gradient_bits)
        self.weight_compressor = WeightCompressor(compression_ratio=0.1)
        self.delta_compressor = DeltaCompressor(threshold=1e-5)
        self.bandwidth_estimator = BandwidthEstimator(
            network_bandwidth_mbps=network_bandwidth_mbps,
            latency_ms=network_latency_ms,
        )
        
        self.round_metrics = {}
        self.cumulative_metrics = {
            "total_bytes_uploaded": 0,
            "total_bytes_downloaded": 0,
            "total_transmission_time_ms": 0.0,
            "total_rounds": 0,
        }
    
    def measure_model_upload_size(
        self,
        model: nn.Module,
        use_quantization: bool = True,
        use_compression: bool = True,
        use_delta: bool = False,
        prev_model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """
        Measure upload size for model parameters with optional compression.
        
        Args:
            model: Model to measure
            use_quantization: Apply gradient quantization
            use_compression: Apply weight compression
            use_delta: Use delta compression (requires prev_model)
            prev_model: Previous model for delta computation
            
        Returns:
            Dictionary with upload metrics
        """
        state_dict = model.state_dict()
        
        # Calculate original size
        original_bytes = sum(
            param.numel() * 4 for param in state_dict.values()
        )
        
        metrics = {
            "original_upload_bytes": int(original_bytes),
        }
        
        current_size = original_bytes
        
        # Apply delta compression if requested
        if use_delta and prev_model is not None:
            prev_state = prev_model.state_dict()
            delta_dict, delta_stats = self.delta_compressor.compute_delta(
                prev_state, state_dict
            )
            metrics["delta_compression"] = delta_stats
            # Estimate compressed size from delta stats
            current_size = delta_stats["compressed_delta_size"] * 4
        
        # Apply weight compression
        if use_compression:
            compressed_dict, comp_stats = self.weight_compressor.compress(model)
            metrics["weight_compression"] = comp_stats
            current_size = comp_stats["compressed_size_bytes"]
        
        # Apply gradient quantization
        if use_quantization:
            grad_quantization_stats = self._estimate_quantization_impact(model)
            metrics["gradient_quantization"] = grad_quantization_stats
            # Quantization reduces size by compression_ratio
            compression_ratio = grad_quantization_stats["compression_ratio"]
            current_size = int(current_size / compression_ratio)
        
        metrics["final_upload_bytes"] = int(current_size)
        metrics["compression_achieved"] = (
            original_bytes / (current_size + 1e-10)
        )
        
        # Estimate transmission time
        transmission_metrics = (
            self.bandwidth_estimator.estimate_bandwidth_savings(
                original_bytes, current_size
            )
        )
        metrics["transmission_metrics"] = transmission_metrics
        
        return metrics
    
    def measure_gradient_quantization(
        self,
        model: nn.Module,
    ) -> Dict[str, Any]:
        """
        Measure impact of gradient quantization on model update sizes.
        
        Args:
            model: Model with gradients computed
            
        Returns:
            Dictionary with quantization metrics
        """
        total_original_bytes = 0
        total_quantized_bytes = 0
        quantized_params = {}
        
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            
            grad = param.grad.data
            original_bytes = grad.numel() * 4  # FP32
            total_original_bytes += original_bytes
            
            quantized_grad, scale, stats = self.gradient_quantizer.quantize(grad)
            total_quantized_bytes += stats["quantized_bytes"]
            
            quantized_params[name] = {
                "scale_factor": stats["scale_factor"],
                "original_bytes": stats["original_bytes"],
                "quantized_bytes": stats["quantized_bytes"],
                "compression_ratio": stats["compression_ratio"],
                "relative_error": stats["relative_quantization_error"],
            }
        
        overall_compression = (
            total_original_bytes / (total_quantized_bytes + 1e-10)
        )
        
        return {
            "total_original_gradient_bytes": int(total_original_bytes),
            "total_quantized_gradient_bytes": int(total_quantized_bytes),
            "overall_compression_ratio": float(overall_compression),
            "quantization_bits": int(self.gradient_bits),
            "num_quantized_params": len(quantized_params),
            "per_parameter_stats": quantized_params,
        }
    
    def record_round_metrics(
        self,
        round_num: int,
        model: nn.Module,
        metrics_dict: Dict[str, Any],
        prev_model: Optional[nn.Module] = None,
    ) -> None:
        """
        Record network metrics for a specific federated round.
        
        Args:
            round_num: Round number
            model: Model after training
            metrics_dict: Additional metrics to include
            prev_model: Previous model for delta compression
        """
        upload_metrics = self.measure_model_upload_size(
            model=model,
            use_quantization=True,
            use_compression=True,
            use_delta=(prev_model is not None),
            prev_model=prev_model,
        )
        
        quant_metrics = self.measure_gradient_quantization(model)
        
        round_data = {
            "round": int(round_num),
            "upload_metrics": upload_metrics,
            "gradient_quantization_metrics": quant_metrics,
            "additional_metrics": metrics_dict,
        }
        
        self.round_metrics[round_num] = round_data
        
        # Update cumulative metrics
        self.cumulative_metrics["total_bytes_uploaded"] += (
            upload_metrics["final_upload_bytes"]
        )
        self.cumulative_metrics["total_transmission_time_ms"] += (
            upload_metrics["transmission_metrics"]["compressed_transmission_time_ms"]
        )
        self.cumulative_metrics["total_rounds"] += 1
    
    def get_round_metrics(self, round_num: int) -> Dict[str, Any]:
        """Get metrics for a specific round."""
        return self.round_metrics.get(round_num, {})
    
    def get_cumulative_metrics(self) -> Dict[str, Any]:
        """Get cumulative network metrics across all rounds."""
        avg_upload_bytes = (
            self.cumulative_metrics["total_bytes_uploaded"] /
            max(self.cumulative_metrics["total_rounds"], 1)
        )
        avg_transmission_time = (
            self.cumulative_metrics["total_transmission_time_ms"] /
            max(self.cumulative_metrics["total_rounds"], 1)
        )
        
        return {
            **self.cumulative_metrics,
            "avg_bytes_per_round": float(avg_upload_bytes),
            "avg_transmission_time_per_round_ms": float(avg_transmission_time),
        }
    
    def _estimate_quantization_impact(
        self,
        model: nn.Module,
    ) -> Dict[str, Any]:
        """Estimate quantization impact on model size."""
        total_params = sum(p.numel() for p in model.parameters())
        
        # FP32 baseline
        fp32_bytes = total_params * 4
        
        # Quantized size (INT8)
        quantized_bytes = total_params * 1
        
        compression_ratio = fp32_bytes / (quantized_bytes + 1e-10)
        
        return {
            "total_parameters": int(total_params),
            "fp32_size_bytes": int(fp32_bytes),
            "quantized_size_bytes": int(quantized_bytes),
            "compression_ratio": float(compression_ratio),
            "quantization_bits": int(self.gradient_bits),
        }


class NetworkCostAnalyzer:
    """Analyzes cumulative network costs and identifies bottlenecks."""
    
    def __init__(self):
        """Initialize network cost analyzer."""
        self.metrics_history: List[Dict[str, Any]] = []
    
    def add_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add metrics to history."""
        self.metrics_history.append(metrics.copy())
    
    def analyze_communication_efficiency(self) -> Dict[str, Any]:
        """
        Analyze communication efficiency across all rounds.
        
        Returns:
            Dictionary with efficiency metrics
        """
        if not self.metrics_history:
            return {}
        
        total_bytes = sum(
            m.get("final_upload_bytes", 0) for m in self.metrics_history
        )
        num_rounds = len(self.metrics_history)
        
        upload_bytes_list = [
            m.get("final_upload_bytes", 0) for m in self.metrics_history
        ]
        compression_ratios = [
            m.get("compression_achieved", 0) for m in self.metrics_history
        ]
        
        avg_upload = sum(upload_bytes_list) / max(num_rounds, 1)
        max_upload = max(upload_bytes_list) if upload_bytes_list else 0
        min_upload = min(upload_bytes_list) if upload_bytes_list else 0
        avg_compression = sum(compression_ratios) / max(num_rounds, 1)
        
        return {
            "total_bytes_transmitted": int(total_bytes),
            "num_rounds": int(num_rounds),
            "avg_bytes_per_round": float(avg_upload),
            "max_bytes_in_round": int(max_upload),
            "min_bytes_in_round": int(min_upload),
            "avg_compression_ratio": float(avg_compression),
        }
    
    def identify_bottlenecks(self) -> Dict[str, Any]:
        """Identify communication bottlenecks."""
        if not self.metrics_history:
            return {}
        
        upload_bytes_list = [
            m.get("final_upload_bytes", 0) for m in self.metrics_history
        ]
        
        # Find rounds with highest communication overhead
        bottleneck_rounds = sorted(
            enumerate(upload_bytes_list),
            key=lambda x: x[1],
            reverse=True
        )[:3]  # Top 3 rounds
        
        return {
            "bottleneck_rounds": [
                {
                    "round": int(idx),
                    "upload_bytes": int(bytes_),
                }
                for idx, bytes_ in bottleneck_rounds
            ],
            "recommendation": (
                "Focus compression techniques on rounds with highest "
                "communication overhead"
            ),
        }


def save_network_metrics_to_file(
    metrics: Dict[str, Any],
    output_path: str = "network_metrics.txt",
) -> None:
    """
    Save network metrics to a human-readable text file.
    
    Args:
        metrics: Dictionary from NetworkMetricsCollector
        output_path: Path to save the metrics file
    """
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("NETWORK METRICS AND COMMUNICATION EFFICIENCY ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Upload Metrics Section
        f.write("UPLOAD SIZE AND COMPRESSION METRICS\n")
        f.write("-" * 80 + "\n")
        upload = metrics.get("upload_metrics", {})
        f.write(
            f"  Original Upload Size: {upload.get('original_upload_bytes', 0) / 1024:.2f} KB\n"
        )
        f.write(
            f"  Final Upload Size: {upload.get('final_upload_bytes', 0) / 1024:.2f} KB\n"
        )
        f.write(
            f"  Compression Achieved: {upload.get('compression_achieved', 0):.2f}x\n\n"
        )
        
        # Transmission Time Section
        f.write("TRANSMISSION TIME ESTIMATES\n")
        f.write("-" * 80 + "\n")
        trans = upload.get("transmission_metrics", {})
        f.write(
            f"  Original Transmission Time: {trans.get('original_transmission_time_ms', 0):.2f} ms\n"
        )
        f.write(
            f"  Compressed Transmission Time: {trans.get('compressed_transmission_time_ms', 0):.2f} ms\n"
        )
        f.write(
            f"  Transmission Time Saved: {trans.get('transmission_time_saved_ms', 0):.2f} ms\n"
        )
        f.write(
            f"  Percentage Saved: {trans.get('percentage_time_saved', 0):.1f}%\n\n"
        )
        
        # Gradient Quantization Section
        f.write("GRADIENT QUANTIZATION METRICS\n")
        f.write("-" * 80 + "\n")
        quant = metrics.get("gradient_quantization_metrics", {})
        f.write(
            f"  Original Gradient Bytes: {quant.get('total_original_gradient_bytes', 0) / 1024:.2f} KB\n"
        )
        f.write(
            f"  Quantized Gradient Bytes: {quant.get('total_quantized_gradient_bytes', 0) / 1024:.2f} KB\n"
        )
        f.write(
            f"  Overall Compression Ratio: {quant.get('overall_compression_ratio', 0):.2f}x\n"
        )
        f.write(
            f"  Quantization Bits: {quant.get('quantization_bits', 0)}\n"
        )
        f.write(
            f"  Number of Quantized Parameters: {quant.get('num_quantized_params', 0)}\n\n"
        )
        
        # Weight Compression Section (if available)
        weight_comp = upload.get("weight_compression", {})
        if weight_comp:
            f.write("WEIGHT COMPRESSION METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"  Original Compressed Size: {weight_comp.get('original_size_bytes', 0) / 1024:.2f} KB\n"
            )
            f.write(
                f"  Final Compressed Size: {weight_comp.get('compressed_size_bytes', 0) / 1024:.2f} KB\n"
            )
            f.write(
                f"  Sparsity: {weight_comp.get('sparsity_percent', 0):.1f}%\n"
            )
            f.write(
                f"  Compression Achieved: {weight_comp.get('compression_achieved', 0):.2f}x\n\n"
            )
        
        # Delta Compression Section (if available)
        delta_comp = upload.get("delta_compression", {})
        if delta_comp:
            f.write("DELTA COMPRESSION METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"  Original Delta Size: {delta_comp.get('original_delta_size', 0)} elements\n"
            )
            f.write(
                f"  Compressed Delta Size: {delta_comp.get('compressed_delta_size', 0)} elements\n"
            )
            f.write(
                f"  Zeros Removed: {delta_comp.get('zero_delta_percent', 0):.1f}%\n\n"
            )
        
        f.write("=" * 80 + "\n")
