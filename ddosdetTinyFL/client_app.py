"""Flower Client Definition for TinyFL Multiclass DDoS Detection for EdgeML and/or Embedded Devices"""

from flwr.clientapp import ClientApp
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from ddosdetTinyFL.task import DDoSClassifier, evaluator, load_data, trainer

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    """Train the Models in partitioned Local Data."""

     # Load Model
    net = DDoSClassifier()
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Load partitioned Dataset
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]

    train_loader, _ = load_data(partition_id, num_partitions, batch_size)

    # Perform Training
    train_loss, duration = trainer(
        net,
        train_loader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
    )

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
    """Evaluate the Model on local partitioned data."""

    # Load Model
    net = DDoSClassifier()
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Load Dataset
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]

    _, test_loader = load_data(partition_id, num_partitions, batch_size)

    # Perform Evaluation
    loss, accuracy, extra_metrics = evaluator(net, test_loader)

    # Construct and return reply Message
    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "f1_score": float(extra_metrics["f1_score"]),
        "precision": float(extra_metrics["precision"]),
        "num-examples": len(test_loader.dataset),
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    
    return Message(content=content, reply_to=msg)

def weighted_average_eval(content: list[RecordDict], server_round:str) -> MetricRecord:
    """Agrega as métricas extraindo o num-examples de dentro de cada registro."""
  
    acc_sum = 0.0
    f1_sum = 0.0
    prec_sum = 0.0
    loss_sum = 0.0
    count = 0
    total_examples = 0

    for c in content:
        metrics = c["metrics"]
        n = metrics["num-examples"]
        
        acc_sum += metrics["accuracy"] * n
        f1_sum += metrics["f1_score"] * n
        prec_sum += metrics["precision"] * n
            
        loss_sum += metrics["loss"] * n

        total_examples += n
        count += 1

    aggregated = {
        "num-examples": total_examples,
    }

    if acc_sum > 0: aggregated["accuracy"] = acc_sum / total_examples
    if f1_sum > 0: aggregated["f1_score"] = f1_sum / total_examples
    if prec_sum > 0: aggregated["precision"] = prec_sum / total_examples
    if loss_sum > 0: aggregated["train_loss"] = loss_sum / total_examples

    return MetricRecord(aggregated)


def weighted_average_train(content: list[RecordDict], server_round:str) -> MetricRecord:
    """Aggregates the metrics by extracting the num-examples from within each record."""

    loss_sum = 0.0
    duration_total = 0.0
    count = 0
    total_examples = 0

    for c in content:
        metrics = c["metrics"]
        n = metrics["num-examples"]
        
        loss_sum += metrics["train_loss"] * n
        duration_total += metrics["train_duration"]
        
        total_examples += n
        count += 1
    
    aggregated = {
        "num-examples": total_examples,
    }

    if loss_sum > 0: aggregated["train_loss"] = loss_sum / total_examples
    if count > 0: aggregated["avg_train_duration"] = duration_total / count

    return MetricRecord(aggregated)