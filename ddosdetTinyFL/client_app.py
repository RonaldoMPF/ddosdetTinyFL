"""Flower Client Definition for TinyFL Multiclass DDoS Detection for EdgeML and/or Embedded Devices"""

from flwr.clientapp import ClientApp
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from ddosdetTinyFL.task import DDoSClassifier, evaluator, load_data, trainer

#import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    """Train the Models in partitioned local data."""

    # Load Model
    net = DDoSClassifier()
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Load partitioned Dataset
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]

    train_loader, _ = load_data(partition_id, num_partitions, batch_size)

    # Perform Training (occurs in FP32 to maintain Gradient Stability)
    train_loss, duration = trainer(
        net,
        train_loader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
    )

    # [TinyML 1] Apply 25% Pruning to Linear Layers after Training
    # This zeroes out the less important Weights before sending them to the Server
    for module in net.modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=0.25)
            prune.remove(module, 'weight') # Fixes the zeros in the Weight Matrix.

    # Construct and return reply Message
    model_record = ArrayRecord(net.state_dict())
    metrics = {
        "train_loss": float(train_loss),
        "train_duration": float(duration),
        "num-examples": len(train_loader.dataset),
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the Model on partitioned local data."""

    # Load Model
    model = DDoSClassifier()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Load Dataset
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]

    _, test_loader = load_data(partition_id, num_partitions, batch_size)

    # [TinyML 1] Reapply Pruning to ensure sparsity during evaluation.
    for module in model.modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=0.25)
            prune.remove(module, 'weight')

    # [TinyML 2] Apply Dynamic Quantization (Converts FP32 -> INT8)
    # Simulates the actual execution of the Optimized Model on the Embedded/Edge Device
    quantize_(model, Int8DynamicActivationInt8WeightConfig())

    # Perform Evaluation using the Quantized Model.
    loss, accuracy, extra_metrics = evaluator(model, test_loader)

    # Construct and return reply Message
    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "f1_score": float(extra_metrics["f1_score"]),
        "precision": float(extra_metrics["precision"]),
        "recall": float(extra_metrics["recall"]),
        "num-examples": len(test_loader.dataset),
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    
    return Message(content=content, reply_to=msg)


def weighted_average_eval(content: list[RecordDict], server_round: str) -> MetricRecord:
    """Aggregates the Evaluation Metrics by extracting the num-examples from within each record."""
    acc_sum, f1_sum, prec_sum, reca_sum, loss_sum = 0.0, 0.0, 0.0, 0.0, 0.0
    total_examples = 0

    for c in content:
        metrics = c["metrics"]
        n = metrics["num-examples"]
        
        acc_sum += metrics["accuracy"] * n
        f1_sum += metrics["f1_score"] * n
        prec_sum += metrics["precision"] * n
        reca_sum += metrics["recall"] * n
        loss_sum += metrics["loss"] * n
        total_examples += n

    aggregated = {"num-examples": total_examples}
    if total_examples > 0:
        aggregated["accuracy"] = acc_sum / total_examples
        aggregated["f1_score"] = f1_sum / total_examples
        aggregated["precision"] = prec_sum / total_examples
        aggregated["recall"] = reca_sum / total_examples
        aggregated["loss"] = loss_sum / total_examples

    return MetricRecord(aggregated)


def weighted_average_train(content: list[RecordDict], server_round: str) -> MetricRecord:
    """Aggregates the Train metrics by extracting the num-examples from within each record."""
    loss_sum, duration_total = 0.0, 0.0
    count, total_examples = 0, 0

    for c in content:
        metrics = c["metrics"]
        n = metrics["num-examples"]
        
        loss_sum += metrics["train_loss"] * n
        duration_total += metrics["train_duration"]
        total_examples += n
        count += 1
    
    aggregated = {"num-examples": total_examples}
    if total_examples > 0:
        aggregated["train_loss"] = loss_sum / total_examples
    if count > 0:
        aggregated["avg_train_duration"] = duration_total / count

    return MetricRecord(aggregated)