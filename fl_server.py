import flwr as fl
from typing import Dict, List, Tuple
import numpy as np


def weighted_avg_metrics(metrics: List[Tuple[int, Dict]]) -> Dict:
    # Aggregate metrics from multiple clients
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    total = sum(num_examples for num_examples, _ in metrics)
    return {
        "accuracy": sum(accuracies) / total,
        "loss": sum(losses) / total
    }


if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_avg_metrics,
        min_fit_clients=2
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )