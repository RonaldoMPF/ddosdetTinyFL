"""Flower Server Definition for TinyFL Multiclass DDoS Detection for EdgeML and/or Embedded Devices"""

import torch
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
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

    # Init global model
    global_model = DDoSClassifier()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggr_fn=weighted_average_eval, # Aggregates Evaluation Metrics.
        train_metrics_aggr_fn=weighted_average_train,   # Aggregates Training metrics.
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Save Final Model to Disk
    print("\nSaving Final Model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    
    torch.save(state_dict, "final-model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data from Dataset."""

    # Load the model and initialize it with the received weights
    global_model = DDoSClassifier()
    global_model.load_state_dict(arrays.to_torch_state_dict())

    # Load entire Test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the Global Model on the Test set
    test_loss, test_acc, extra_metrics = evaluator(global_model, test_dataloader)

    # Return the Evaluation Metrics
    return MetricRecord({
        "accuracy": test_acc, 
        "loss": test_loss,
        "f1_score": extra_metrics["f1_score"],
        "precision": extra_metrics["precision"]
    })