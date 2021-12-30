import flwr as fl
from flwr.server import strategy
import flwr as fl

customStrat = fl.server.strategy.FedAvg(
    #fraction_fit=0.1, #Sample 10% of available clients for the next round 
    min_eval_clients = 10,
    min_fit_clients=10, #Minimum numer of clients to be sampled for next round
    min_available_clients=10  #minimum number of clients that need to connect before start
)
fl.server.start_server(config={"num_rounds": 10}, strategy=customStrat)
