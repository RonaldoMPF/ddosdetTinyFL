"""Metrics Collection and Export Utilities for Federated Learning Experiments.

This module provides functionality to collect, aggregate, and export metrics from
both standard and optimized model scenarios in formats suitable for plotting and analysis.
"""

import json
import csv
from typing import Dict, Any, List, Optional
from pathlib import Path
import statistics
import time


class MetricsCollector:
    """Collects and aggregates metrics from federated learning rounds."""
    
    def __init__(self, scenario_name: str = "default"):
        """Initialize metrics collector for a specific scenario.
        
        Args:
            scenario_name: Name of the scenario (e.g., "standard", "optimized")
        """
        self.scenario_name = scenario_name
        self.round_metrics: Dict[int, Dict[str, Any]] = {}
        self.client_metrics: Dict[int, List[Dict[str, Any]]] = {}  # Per-round client metrics
        self.global_metrics: Dict[str, Any] = {}
        self.start_time = time.time()
    
    def record_round_metrics(
        self,
        round_num: int,
        metrics: Dict[str, Any],
        client_count: int = 0
    ) -> None:
        """Record aggregated metrics for a federated learning round.
        
        Args:
            round_num: The federated learning round number
            metrics: Dictionary of aggregated metrics for the round
            client_count: Number of clients that participated
        """
        self.round_metrics[round_num] = {
            'round': round_num,
            'timestamp': time.time() - self.start_time,
            'client_count': client_count,
            **metrics
        }
    
    def record_client_metrics(
        self,
        round_num: int,
        client_id: int,
        metrics: Dict[str, Any]
    ) -> None:
        """Record individual client metrics for a round.
        
        Args:
            round_num: The federated learning round number
            client_id: Client identifier
            metrics: Dictionary of client-specific metrics
        """
        if round_num not in self.client_metrics:
            self.client_metrics[round_num] = []
        
        self.client_metrics[round_num].append({
            'client_id': client_id,
            **metrics
        })
    
    def record_global_metrics(self, metrics: Dict[str, Any]) -> None:
        """Record final global model evaluation metrics.
        
        Args:
            metrics: Dictionary of final global metrics
        """
        self.global_metrics = {
            'scenario': self.scenario_name,
            'timestamp': time.time() - self.start_time,
            **metrics
        }
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics across all rounds.
        
        Returns:
            Dictionary containing aggregated metrics across rounds
        """
        if not self.round_metrics:
            return {}
        
        summary = {
            'scenario': self.scenario_name,
            'total_rounds': len(self.round_metrics),
            'total_duration_sec': time.time() - self.start_time,
        }
        
        # Extract all metric keys from first round
        first_round = self.round_metrics[min(self.round_metrics.keys())]
        metric_keys = [k for k in first_round.keys() 
                      if k not in ['round', 'timestamp', 'client_count']]
        
        # Calculate statistics for each metric
        for key in metric_keys:
            values = [
                self.round_metrics[r].get(key)
                for r in sorted(self.round_metrics.keys())
                if self.round_metrics[r].get(key) is not None
            ]
            
            if values and isinstance(values[0], (int, float)):
                summary[f'{key}_mean'] = statistics.mean(values)
                summary[f'{key}_std'] = statistics.stdev(values) if len(values) > 1 else 0
                summary[f'{key}_min'] = min(values)
                summary[f'{key}_max'] = max(values)
        
        return summary
    
    def export_to_json(self, output_path: str) -> None:
        """Export collected metrics to JSON format.
        
        Args:
            output_path: Path to save the JSON metrics file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'scenario': self.scenario_name,
            'round_metrics': self.round_metrics,
            'global_metrics': self.global_metrics,
            'summary_statistics': self.get_summary_statistics(),
            'collection_time_sec': time.time() - self.start_time,
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def export_to_csv(self, output_path: str) -> None:
        """Export metrics to CSV format suitable for plotting.
        
        Args:
            output_path: Path to save the CSV metrics file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if not self.round_metrics:
            return
        
        # Get all metric keys from all rounds
        all_keys = set()
        for metrics in self.round_metrics.values():
            all_keys.update(metrics.keys())
        
        # Sort keys for consistent column ordering
        fieldnames = sorted(all_keys)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for round_num in sorted(self.round_metrics.keys()):
                writer.writerow(self.round_metrics[round_num])
    
    def export_per_round_to_csv(self, output_dir: str) -> None:
        """Export per-round metrics to separate CSV files for advanced analysis.
        
        Args:
            output_dir: Directory to save per-round CSV files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for round_num, metrics in self.round_metrics.items():
            round_file = output_path / f"round_{round_num:03d}.csv"
            
            with open(round_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=metrics.keys())
                writer.writeheader()
                writer.writerow(metrics)


class ComparativeMetricsAnalyzer:
    """Analyzes and compares metrics between two scenarios (standard vs. optimized)."""
    
    def __init__(
        self,
        standard_collector: MetricsCollector,
        optimized_collector: MetricsCollector
    ):
        """Initialize comparative analyzer with two scenario collectors.
        
        Args:
            standard_collector: MetricsCollector from standard scenario
            optimized_collector: MetricsCollector from optimized scenario
        """
        self.standard = standard_collector
        self.optimized = optimized_collector
    
    def compute_delta_metrics(self) -> Dict[str, Any]:
        """Compute delta (difference) metrics between scenarios.
        
        Returns:
            Dictionary containing differences and improvement percentages
        """
        std_global = self.standard.global_metrics
        opt_global = self.optimized.global_metrics
        
        deltas = {'scenario_comparison': 'standard_vs_optimized'}
        
        # Compare common metrics
        for key in std_global:
            if key in opt_global and key not in ['scenario', 'timestamp']:
                std_val = std_global[key]
                opt_val = opt_global[key]
                
                if isinstance(std_val, (int, float)) and isinstance(opt_val, (int, float)):
                    # For accuracy and similar metrics: higher is better
                    if key in ['accuracy', 'f1_score', 'precision', 'recall']:
                        diff = opt_val - std_val
                        pct_change = (diff / std_val * 100) if std_val != 0 else 0
                    else:
                        # For loss and latency: lower is better
                        diff = std_val - opt_val
                        pct_change = (diff / std_val * 100) if std_val != 0 else 0
                    
                    deltas[f'{key}_delta'] = diff
                    deltas[f'{key}_pct_change'] = pct_change
        
        return deltas
    
    def export_comparison_to_json(self, output_path: str) -> None:
        """Export comparative analysis to JSON.
        
        Args:
            output_path: Path to save the comparison JSON file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        comparison = {
            'standard_scenario': {
                'round_metrics': self.standard.round_metrics,
                'global_metrics': self.standard.global_metrics,
                'summary': self.standard.get_summary_statistics(),
            },
            'optimized_scenario': {
                'round_metrics': self.optimized.round_metrics,
                'global_metrics': self.optimized.global_metrics,
                'summary': self.optimized.get_summary_statistics(),
            },
            'delta_metrics': self.compute_delta_metrics(),
        }
        
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
    
    def export_comparison_to_csv(self, output_path: str) -> None:
        """Export comparative metrics to CSV in a format suitable for plotting.
        
        Args:
            output_path: Path to save the comparison CSV file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Collect all rounds from both scenarios
        all_rounds = set(
            list(self.standard.round_metrics.keys()) +
            list(self.optimized.round_metrics.keys())
        )
        
        rows = []
        for round_num in sorted(all_rounds):
            row = {'round': round_num}
            
            # Add standard metrics
            if round_num in self.standard.round_metrics:
                std_metrics = self.standard.round_metrics[round_num]
                for key, value in std_metrics.items():
                    if key not in ['round', 'timestamp', 'client_count']:
                        row[f'standard_{key}'] = value
            
            # Add optimized metrics
            if round_num in self.optimized.round_metrics:
                opt_metrics = self.optimized.round_metrics[round_num]
                for key, value in opt_metrics.items():
                    if key not in ['round', 'timestamp', 'client_count']:
                        row[f'optimized_{key}'] = value
            
            rows.append(row)
        
        if rows:
            fieldnames = sorted(set().union(*(r.keys() for r in rows)))
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    
    def generate_comparison_report(self, output_path: str) -> None:
        """Generate a human-readable comparison report.
        
        Args:
            output_path: Path to save the report text file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FEDERATED LEARNING EXPERIMENT - SCENARIO COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Standard Scenario Summary
            f.write("STANDARD SCENARIO (FP32 Model without Optimization)\n")
            f.write("-" * 80 + "\n")
            std_summary = self.standard.get_summary_statistics()
            for key, value in std_summary.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Optimized Scenario Summary
            f.write("OPTIMIZED SCENARIO (Pruned + Quantized Model)\n")
            f.write("-" * 80 + "\n")
            opt_summary = self.optimized.get_summary_statistics()
            for key, value in opt_summary.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Delta Metrics
            f.write("COMPARISON (Optimized vs. Standard)\n")
            f.write("-" * 80 + "\n")
            delta = self.compute_delta_metrics()
            for key, value in delta.items():
                if key != 'scenario_comparison':
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
            
            f.write("=" * 80 + "\n")


def create_metrics_comparison_plot_data(
    standard_path: str,
    optimized_path: str,
    output_path: str
) -> None:
    """Load metrics from two JSON files and create comparison data for plotting.
    
    Args:
        standard_path: Path to standard scenario metrics JSON
        optimized_path: Path to optimized scenario metrics JSON
        output_path: Path to save comparison data JSON
    """
    with open(standard_path, 'r') as f:
        standard_data = json.load(f)
    
    with open(optimized_path, 'r') as f:
        optimized_data = json.load(f)
    
    # Prepare comparison data
    comparison_data = {
        'scenarios': ['standard', 'optimized'],
        'round_metrics': {
            'standard': standard_data.get('round_metrics', {}),
            'optimized': optimized_data.get('round_metrics', {}),
        },
        'global_metrics': {
            'standard': standard_data.get('global_metrics', {}),
            'optimized': optimized_data.get('global_metrics', {}),
        },
        'summary_statistics': {
            'standard': standard_data.get('summary_statistics', {}),
            'optimized': optimized_data.get('summary_statistics', {}),
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
