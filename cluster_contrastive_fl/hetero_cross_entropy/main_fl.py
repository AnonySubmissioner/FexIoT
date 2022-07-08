## We appreciate the code framework provided by "Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints"
## Part of the code is reused from: https://github.com/felisat/clustered-federated-learning

import torch, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torchvision import datasets, transforms
from copy import deepcopy

from helper import ExperimentLogger, display_train_stats
from fl_devices import Server, Client
from data_utils import split_noniid, CustomSubset

import sys
sys.path.append("..")
from heterograph_dataset.build_heter_graph import HeteroGraphDataset
from magnn import GraphClassifier

torch.manual_seed(42)
np.random.seed(42)

N_CLIENTS = 3
DIRICHLET_ALPHA = 0.5
data = HeteroGraphDataset()

idcs = np.random.permutation(len(data))
train_idcs, test_idcs = idcs[:int(len(data)*0.8)], idcs[int(len(data)*0.8):]
train_labels = data.labels.numpy()

client_idcs = split_noniid(train_idcs, train_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)

client_data = [CustomSubset(data, idcs) for idcs in client_idcs]
test_data = CustomSubset(data, test_idcs)

clients = [Client(GraphClassifier, lambda x : torch.optim.Adam(x, lr=0.0007, weight_decay=1e-5), dat, idnum=i) 
           for i, dat in enumerate(client_data)]
server = Server(GraphClassifier, test_data)

COMMUNICATION_ROUNDS = 5
EPS_1 = 1.5
EPS_2 = 1.7


cfl_stats = ExperimentLogger()
    
cluster_indices = [np.arange(len(clients)).astype("int")]
client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

cluster_time = 0

for c_round in range(1, COMMUNICATION_ROUNDS+1):

    if c_round == 1:
        for client in clients:
            client.synchronize_with_server(server)
            
    participating_clients = server.select_clients(clients, frac=1)

    for client in participating_clients:
        train_stats = client.compute_weight_update(epochs=10)
        client.reset()

    similarities = server.compute_pairwise_similarities(clients, True, cluster_time)

    cluster_indices_new = []
    for idc in cluster_indices:
        max_norm = server.compute_max_update_norm([clients[i] for i in idc])
        mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
             
        if mean_norm<EPS_1 and max_norm>EPS_2 and len(idc)>2 and c_round>10 and cluster_time<8:
            
            # server.cache_model(idc, clients[idc[0]].W, acc_clients)
            c1, c2 = server.cluster_clients(similarities[idc][:,idc], idc)
            if len(c1) == 0 or len(c2) == 0:
                cluster_indices_new += [idc]
                continue
            cluster_time+=1
            cluster_indices_new += [c1, c2]
             
            cfl_stats.log({"split" : c_round})

        else:
            cluster_indices_new += [idc]
            
    cluster_indices = cluster_indices_new
    
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    server.aggregate_clusterwise(client_clusters)

    
    acc_clients = []
    prfs_clients = []
    server_agg_clients= []
    client_syn_server = []
    for client in clients:
        acc, prfs= client.evaluate()
        acc_clients.append(acc)
        prfs_clients.append(prfs)
        if acc > 0.5:
            server_agg_clients.append(client)
        else:
            client_syn_server.append(client)
    
    if c_round > 50:       
        server.aggregate_weight_updates(server_agg_clients)

    cfl_stats.log({"acc_clients" : acc_clients, "mean_norm" : mean_norm, "max_norm" : max_norm,
                  "rounds" : c_round, "clusters" : cluster_indices})
    
    
    display_train_stats(cfl_stats, EPS_1, EPS_2, COMMUNICATION_ROUNDS)

    
for idc in cluster_indices:    
    server.cache_model(idc, clients[idc[0]].W, acc_clients)

results = np.zeros([N_CLIENTS, len(server.model_cache)])

for i, (idcs, W, accs) in enumerate(server.model_cache):
    results[idcs, i] = np.array(accs)

n_clusters = len(cluster_indices)
print(np.mean(cfl_stats.acc_clients, axis=1), np.std(cfl_stats.acc_clients, axis=1))
