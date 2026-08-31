import flwr as fl


class FederatedAveraging(fl.server.strategy.FedAvg):
    """Standard weighted FedAvg; secure aggregation is not enabled here."""


if __name__ == "__main__":
    fl.server.start_server(
        server_address="127.0.0.1:9090",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=FederatedAveraging(
            on_fit_config_fn=lambda server_round: {"server_round": server_round}
        ),
    )
