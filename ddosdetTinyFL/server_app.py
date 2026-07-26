"""Flower Server Definition for TinyFL Multiclass DDoS Detection for EdgeML and/or Embedded Devices"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig

from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedProx
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from ddosdetTinyFL.task import DDoSClassifier, load_centralized_dataset, evaluator
from ddosdetTinyFL.client_app import weighted_average_train, weighted_average_eval

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Init Global Model
    global_model = DDoSClassifier()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedProx strategy
    # FedProx tolerates the heterogeneity caused by Local Client Pruning
    strategy = FedProx(
        proximal_mu=1.0,
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggr_fn=weighted_average_eval, # Aggregates Evaluation Metrics
        train_metrics_aggr_fn=weighted_average_train,   # Aggregates Training Metrics
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        evaluate_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Saving the Final Models
    print("\n[TinyFL Multiclass DDoS Detection] Processing and saving the Final Models...")
    state_dict_final = result.arrays.to_torch_state_dict()
    
    # 1. Save the Standard Global Version (FP32) in case you want to resume Federated Training later
    base_model = DDoSClassifier()
    base_model.load_state_dict(state_dict_final)
    torch.save(base_model.state_dict(), "final-model-fp32.pt")
    
    # 2. Generates and saves the Optimized Version for Edge/Embedded Devices (Pruned + Quantized)
    for module in base_model.modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=0.25)
            prune.remove(module, 'weight')
            
    quantize_(base_model, Int8DynamicActivationInt8WeightConfig())
    
    # Generates and saves the Optimized Version for Edge/Embedded (Pruned + Quantized)
    torch.save(base_model.state_dict(), "final-model-edge-int8.pt")
    print("[SUCCESS] Models Saved:")
    print("  -> 'final-model-fp32.pt' (Base Model)")
    print("  -> 'final-model-edge-int8.pt' (Optimized Model ready for Edge/Embedded Devices)")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate Model on central data from Dataset applying Edge optimizations."""

    # Load the Model and initialize it with the received Weights (FP32)
    global_model = DDoSClassifier()
    global_model.load_state_dict(arrays.to_torch_state_dict())

    # [TinyML] Applies the same Client modifications for a fair evaluation
    # 1. 25% Pruning
    for module in global_model.modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=0.25)
            prune.remove(module, 'weight')

    # 2. Dynamic Quantization to INT8
    quantize_(global_model, Int8DynamicActivationInt8WeightConfig())

    # Load entire Test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the Optimized Global Model on the Test set
    test_loss, test_acc, extra_metrics = evaluator(global_model, test_dataloader)

    # Return the Evaluation Metrics
    return MetricRecord({
        "accuracy": test_acc, 
        "loss": test_loss,
        "f1_score": extra_metrics["f1_score"],
        "precision": extra_metrics["precision"],
        "recall": extra_metrics["recall"]
    })