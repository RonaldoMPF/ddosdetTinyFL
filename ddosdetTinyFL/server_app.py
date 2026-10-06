"""Flower Server Definition for TinyFL Multiclass DDoS Detection for EdgeML and/or Embedded Devices.

This server uses FedAvg and keeps the experimental variables centralized via
`ddosdetTinyFL.experimental_config.ExperimentalConfig`.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from ddosdetTinyFL.client_app import weighted_average_eval, weighted_average_train
from ddosdetTinyFL.experimental_config import get_default_config
from ddosdetTinyFL.task import DDoSClassifier, evaluator, load_centralized_dataset

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    config = get_default_config()
    fraction_evaluate: float = context.run_config.get("fraction-evaluate", config.fraction_evaluate)
    num_rounds: int = context.run_config.get("num-server-rounds", config.num_server_rounds)
    lr: float = context.run_config.get("learning-rate", config.learning_rate)

    # Init Global Model
    global_model = DDoSClassifier()
    arrays = ArrayRecord(global_model.state_dict())

    # Use FedAvg instead of FedProx
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=1,
        min_evaluate_clients=1,
        evaluate_metrics_aggregation_fn=weighted_average_eval,
        fit_metrics_aggregation_fn=weighted_average_train,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        evaluate_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    print("\n[TinyFL Multiclass DDoS Detection] Processing and saving the final models...")
    state_dict_final = result.arrays.to_torch_state_dict()

    # 1. Save the standard global model (FP32)
    base_model = DDoSClassifier()
    base_model.load_state_dict(state_dict_final)
    torch.save(base_model.state_dict(), "final-model-fp32.pt")

    # 2. Save the optimized edge model (pruned + quantized)
    optimized_model = DDoSClassifier()
    optimized_model.load_state_dict(state_dict_final)
    for module in optimized_model.modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=config.prune_ratio)
            prune.remove(module, "weight")
    quantize_(optimized_model, Int8DynamicActivationInt8WeightConfig())
    torch.save(optimized_model.state_dict(), "final-model-edge-int8.pt")

    print("[SUCCESS] Models saved:")
    print("  -> 'final-model-fp32.pt' (Base model)")
    print("  -> 'final-model-edge-int8.pt' (Optimized model ready for edge/embedded devices)")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate the global model using the centralized test set.

    This is the comparison evaluation point. The repository can keep the same
    metrics as before while the strategy is now FedAvg.
    """

    global_model = DDoSClassifier()
    global_model.load_state_dict(arrays.to_torch_state_dict())

    # Apply optimization for the embedded version during server-side evaluation
    for module in global_model.modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=0.25)
            prune.remove(module, "weight")
    quantize_(global_model, Int8DynamicActivationInt8WeightConfig())

    test_dataloader = load_centralized_dataset()
    test_loss, test_acc, extra_metrics = evaluator(global_model, test_dataloader)

    return MetricRecord(
        {
            "accuracy": test_acc,
            "loss": test_loss,
            "f1_score": extra_metrics["f1_score"],
            "precision": extra_metrics["precision"],
            "recall": extra_metrics["recall"],
        }
    )
