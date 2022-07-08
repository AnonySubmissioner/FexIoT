import os
import torch, time
import numpy as np
import pandas as pd
import os.path as osp
from pyg_gcn_ifttt import GCN
from dig.xgraph.models import *
from dig.xgraph.method import SubgraphX
from dig.xgraph.method import GNNExplainer, PGExplainer
from dig.xgraph.evaluation import XCollector
from torch_geometric.data import DataLoader
from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.evaluation import XCollector
from torch.utils.data.sampler import SubsetRandomSampler
from dig.xgraph.method.subgraphx import PlotUtils
from dig.xgraph.method.subgraphx import find_closest_node_result
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip

import sys
sys.path.append(".")
from ifttt_build_dataset.pyg_build_ifttt_graph_dataset import IFTTTDataset, BatchNewDimData

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

IFTTTDataset = IFTTTDataset('./ifttt_build_dataset')
# dataset = dataset.to(device)
num_classes = 2
graph_id = 517

def get_features(name, features):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

model = GCN(300, 64, 2)
features={}
model.lin1.register_forward_hook(get_features('lin1_feats', features))
model = torch.load('./pyg_gcn_ifttt.pt')
model.eval()


explainer = SubgraphX(model, num_classes=2, device=device,
                      explain_graph=True, reward_method='mc_l_shapley',  subgraph_building_method='split')

explainer2 = SubgraphX(model, num_classes=2, device=device,
                      explain_graph=True, reward_method='mc_l_SHAP',  subgraph_building_method='split')

explainer3 = SubgraphX(model, num_classes=2, device=device,
                      explain_graph=True, reward_method='gnn_score',  subgraph_building_method='split')

explainer4 = GNNExplainer(model, epochs=1, explain_graph=True, )

explainer5 = PGExplainer(model, 10,device='cpu')

action_dict = pd.read_csv('./ifttt_build_dataset/action.csv', header=0, index_col=0, squeeze=True).to_dict()
trigger_dict = pd.read_csv('./ifttt_build_dataset/trigger.csv', header=0, index_col=0, squeeze=True).to_dict()
trigger_action_dict = {**trigger_dict, **action_dict}

cnt = 0 
for i in range(len(IFTTTDataset)):
    data = IFTTTDataset.get(i)
    # if np.shape(data.edge_index)[1] >10 and data.y == 1: #517
    if np.shape(data.edge_index)[1] >10 and data.y == 1 and i ==606:
        print(i)
        cnt+=1
        for i in range(len(data.edge_index[0])):
            first = int(data.edge_index[0][i])
            second = int(data.edge_index[1][i])
            print(first, trigger_action_dict[first], "-->", second, trigger_action_dict[second])
        if cnt > 1:
            break

cnt = 0 
idx_list=[]
for i in range(len(IFTTTDataset)):
    data = IFTTTDataset.get(i)
    # if np.shape(data.edge_index)[1] >10 and data.y == 1: #517
    if np.shape(data.edge_index)[1] >10 and data.y == 1:
        # print(i)
        cnt+=1
        idx_list.append(i)
        for i in range(len(data.edge_index[0])):
            first = int(data.edge_index[0][i])
            second = int(data.edge_index[1][i])          
        if cnt > 100:
            break

data = IFTTTDataset.get(graph_id) # 606, 1446, 517
a = model(data.x, data.edge_index)

# --- Create data collector and explanation processor ---

x_collector = XCollector()

max_nodes = 50


for id in idx_list:
    data = IFTTTDataset.get(id) # 606, 1446, 517
    a = model(data.x, data.edge_index)
    print(a)

    labels = torch.tensor([data.y], dtype=torch.int8)
    prediction = model(data.x, data.edge_index).argmax(1)
    _, explanation_results, related_preds = \
    explainer(data.x, data.edge_index, max_nodes=max_nodes)
    explanation_results = explanation_results[prediction]
    if len(explanation_results[0]['coalition'])>20 and len(explanation_results[1]['coalition'])>20:
        print(id)
    else:
        if len(explanation_results[0]['coalition'])<20:
            print(explanation_results[0]['coalition'])
        else:
            print(explanation_results[1]['coalition'])
    explanation_results = explainer.read_from_MCTSInfo_list(explanation_results)


logits = model(data.x, data.edge_index)
prediction = logits.argmax(-1).item()

_, explanation_results, related_preds = \
    explainer(data.x, data.edge_index, max_nodes=max_nodes)

explanation_results = explanation_results[prediction]
explanation_results = explainer.read_from_MCTSInfo_list(explanation_results)
result = find_closest_node_result(explanation_results, max_nodes=max_nodes)

x_collector.collect_data(result.coalition, related_preds, label=prediction)


print(f'Fidelity: {x_collector.fidelity:.4f}\n',
      f'Infidelity: {x_collector.fidelity_inv:.4f}\n'
      f'Sparsity: {x_collector.sparsity:.4f}')


_, explanation_results, related_preds = \
    explainer2(data.x, data.edge_index, max_nodes=max_nodes)

explanation_results = explanation_results[prediction]
explanation_results = explainer2.read_from_MCTSInfo_list(explanation_results)
result = find_closest_node_result(explanation_results, max_nodes=max_nodes)

x_collector.collect_data(result.coalition, related_preds, label=prediction)
print(f'Fidelity: {x_collector.fidelity:.4f}\n',
      f'Infidelity: {x_collector.fidelity_inv:.4f}\n'
      f'Sparsity: {x_collector.sparsity:.4f}')


_, explanation_results, related_preds = \
    explainer3(data.x, data.edge_index, max_nodes=max_nodes)

explanation_results = explanation_results[prediction]
explanation_results = explainer3.read_from_MCTSInfo_list(explanation_results)
result = find_closest_node_result(explanation_results, max_nodes=max_nodes)

x_collector.collect_data(result.coalition, related_preds, label=prediction)
print(f'Fidelity: {x_collector.fidelity:.4f}\n',
      f'Infidelity: {x_collector.fidelity_inv:.4f}\n'
      f'Sparsity: {x_collector.sparsity:.4f}')

cnt =0
idx_list=[]
for i in range(len(IFTTTDataset)):
    data = IFTTTDataset.get(i)
    # if np.shape(data.edge_index)[1] >10 and data.y == 1: #517
    if np.shape(data.edge_index)[1] >5 and data.y == 1:
        idx_list.append(i)
        cnt+=1
    if cnt >49:
        break

sx_inter = [] 
fy_inter = []
sx_subgr = []
fy_subgr = []
sx_score = []
fy_score = []

for idx in idx_list:
    data = IFTTTDataset.get(idx) # 606, 1446, 517
    a = model(data.x, data.edge_index)
    # print(a)

    labels = torch.tensor([data.y], dtype=torch.int8)
    prediction = model(data.x, data.edge_index).argmax(1)
    _, explanation_results, related_preds = \
    explainer(data.x, data.edge_index, max_nodes=max_nodes)
    explanation_results = explanation_results[prediction]
    # print(explanation_results[0])
    explanation_results = explainer.read_from_MCTSInfo_list(explanation_results)


    _2, explanation_results2, related_preds2 = \
    explainer2(data.x, data.edge_index, max_nodes=max_nodes)
    explanation_results2 = explanation_results2[prediction]
    # print(explanation_results2[0])
    explanation_results2 = explainer2.read_from_MCTSInfo_list(explanation_results2)


    _3, explanation_results3, related_preds3 = \
    explainer3(data.x, data.edge_index, max_nodes=max_nodes)
    explanation_results3 = explanation_results3[prediction]
    # print(explanation_results3[0])
    explanation_results3 = explainer3.read_from_MCTSInfo_list(explanation_results3)

    logits = model(data.x, data.edge_index)
    prediction = logits.argmax(-1).item()

    _, explanation_results, related_preds = \
        explainer(data.x, data.edge_index, max_nodes=max_nodes)

    explanation_results = explanation_results[prediction]
    explanation_results = explainer.read_from_MCTSInfo_list(explanation_results)
    result = find_closest_node_result(explanation_results, max_nodes=max_nodes)

    x_collector.collect_data(result.coalition, related_preds, label=prediction)


#     print(f'Fidelity: {x_collector.fidelity:.4f}\n',
#           f'Infidelity: {x_collector.fidelity_inv:.4f}\n'
#           f'Sparsity: {x_collector.sparsity:.4f}')
    
    sx_subgr.append(x_collector.sparsity)
    fy_subgr.append(x_collector.fidelity)
    
    _, explanation_results, related_preds = \
        explainer2(data.x, data.edge_index, max_nodes=max_nodes)

    explanation_results = explanation_results[prediction]
    explanation_results = explainer2.read_from_MCTSInfo_list(explanation_results)
    result = find_closest_node_result(explanation_results, max_nodes=max_nodes)

    x_collector.collect_data(result.coalition, related_preds, label=prediction)
    # print(f'Fidelity: {x_collector.fidelity:.4f}\n',
    #       f'Infidelity: {x_collector.fidelity_inv:.4f}\n'
    #       f'Sparsity: {x_collector.sparsity:.4f}')
    

    sx_inter.append(x_collector.sparsity)
    fy_inter.append(x_collector.fidelity)

    _, explanation_results, related_preds = \
        explainer3(data.x, data.edge_index, max_nodes=max_nodes)

    explanation_results = explanation_results[prediction]
    explanation_results = explainer3.read_from_MCTSInfo_list(explanation_results)
    result = find_closest_node_result(explanation_results, max_nodes=max_nodes)

    x_collector.collect_data(result.coalition, related_preds, label=prediction)
    # print(f'Fidelity: {x_collector.fidelity:.4f}\n',
    #       f'Infidelity: {x_collector.fidelity_inv:.4f}\n'
    #       f'Sparsity: {x_collector.sparsity:.4f}')
    
    sx_score.append(x_collector.sparsity)
    fy_score.append(x_collector.fidelity)
    
print(sx_subgr, fy_subgr, sx_inter, fy_inter, sx_score, fy_score)


cnt =0
idx_list=[]
node_numbers=0
for i in range(len(IFTTTDataset)):
    data = IFTTTDataset.get(i)
    # if np.shape(data.edge_index)[1] >10 and data.y == 1: #517
    if np.shape(data.edge_index)[1] >10 and data.y == 1:
        node_numbers+=np.shape(data.edge_index)[1]
        idx_list.append(i)
        cnt+=1
    if cnt >49:
        break

sx_inter = [] 
fy_inter = []
sx_subgr = []
fy_subgr = []
sx_score = []
fy_score = []

print("The number of graphs: ", len(idx_list))
print("Average node: ", node_numbers/len(idx_list))
start=time.time()
for idx in idx_list:
    data = IFTTTDataset.get(idx) # 606, 1446, 517
    a = model(data.x, data.edge_index)
    # print(a)

    _2, explanation_results2, related_preds2 = \
    explainer2(data.x, data.edge_index, max_nodes=max_nodes)
    explanation_results2 = explanation_results2[prediction]
    # print(explanation_results2[0])
    explanation_results2 = explainer2.read_from_MCTSInfo_list(explanation_results2)
print("The cost time: ", time.time()-start)