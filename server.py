import flwr as fl

#0.0.0.0 as client and server both on the same machine
fl.server.start_server(server_address="0.0.0.0:8000",
                       config=fl.server.ServerConfig(num_rounds=3),
                       strategy=fl.server.strategy.FedAvg)

