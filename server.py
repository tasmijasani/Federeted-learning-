import flwr as fl
from flwr.server.strategy import FedAvg

# Define the strategy for federated learning
strategy = FedAvg(
    fraction_fit=1.0,  # Sample all available clients for training
    fraction_evaluate=1.0,  # Sample all available clients for evaluation
    min_fit_clients=1,  # Minimum number of clients to be sampled for training
    min_evaluate_clients=1,  # Minimum number of clients to be sampled for evaluation
    min_available_clients=1,  # Minimum number of clients that need to be connected to the server
)

# Start the Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)
