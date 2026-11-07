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
from ddosdetTinyFL.experimental_config import ExperimentalConfig, get_default_config
from ddosdetTinyFL.task import DDoSClassifier, evaluator, load_centralized_dataset

# Create ServerApp
app = ServerApp()


def _optimize_model(model: nn.Module, config: ExperimentalConfig) -> nn.Module:
    """Apply the configured pruning and quantization steps to a model."""
    if config.enable_pruning:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(
                    module, name="weight", amount=config.pruning_amount
                )
                prune.remove(module, "weight")

    if config.enable_quantization:
        quantize_(model, Int8DynamicActivationInt8WeightConfig())

    return model


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
        fraction_train=1.0,
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggr_fn=weighted_average_eval,
        train_metrics_aggr_fn=weighted_average_train,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        evaluate_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=lambda server_round, arrays: global_evaluate(
            server_round, arrays, config
        ),
    )

    print("\n[TinyFL Multiclass DDoS Detection] Processing and saving the final models...")
    state_dict_final = result.arrays.to_torch_state_dict()

    # 1. Save the standard global model (FP32)
    base_model = DDoSClassifier()
    base_model.load_state_dict(state_dict_final)
    torch.save(base_model.state_dict(), "final-model-fp32.pt")

    # 2. Save the optimized edge model using the configured optimization settings.
    optimized_model = DDoSClassifier()
    optimized_model.load_state_dict(state_dict_final)
    _optimize_model(optimized_model, config)
    torch.save(optimized_model.state_dict(), "final-model-edge-int8.pt")

    print("[SUCCESS] Models saved:")
    print("  -> 'final-model-fp32.pt' (Base model)")
    print("  -> 'final-model-edge-int8.pt' (Optimized model ready for edge/embedded devices)")


def global_evaluate(
    server_round: int,
    arrays: ArrayRecord,
    config: ExperimentalConfig | None = None,
) -> MetricRecord:
    """Evaluate the configured optimized global model using the centralized test set."""
    config = config or get_default_config()
    global_model = DDoSClassifier()
    global_model.load_state_dict(arrays.to_torch_state_dict())
    _optimize_model(global_model, config)

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
