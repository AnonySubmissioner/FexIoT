import random
import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.cluster import AgglomerativeClustering, Birch, AffinityPropagation
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


device = "cuda:7" if torch.cuda.is_available() else "cpu"

def contrastive_loss(logits_embedding: Tensor, labels: Tensor):
    loss = 0
    len_embedding = len(logits_embedding)

    dis_matrix = torch.cdist(logits_embedding, logits_embedding)

    for idx1 in range(len_embedding):
        for idx2 in range(len_embedding):
            if labels[idx1] == labels[idx2]:
                y = 0
            else:
                y = 1
            dd = dis_matrix[idx1][idx2]
            loss+=(1-y)*dd*dd+y*(60-dd)*(60-dd)
    loss /=(len_embedding*len_embedding)

    return loss


def train_op(model, loader, optimizer, epochs=1):
    model.train()  
    for ep in range(epochs):
        running_loss, samples = 0.0, 0
        for x, y in loader: 
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = contrastive_loss(model(x), y)
            running_loss += loss.item()*y.shape[0]
            samples += y.shape[0]
            loss.backward()
            optimizer.step()  
    return running_loss / samples


def eval_op(client_id, model, train_loader, eval_loader):
    model.eval()
    train_rep_list = []
    train_y_list = []
    with torch.no_grad():
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            representation = model(x).cpu().detach().numpy()
            y = y.numpy()
            train_rep_list.append(representation)
            train_y_list.append(y)
    train_rep = np.concatenate(train_rep_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)
    clf = SGDClassifier(max_iter=1000, tol=1e-3)
    clf.fit(train_rep, train_y)
    
    test_rep_list = []
    test_y_list = []
    with torch.no_grad():
        for i, (x, y) in enumerate(eval_loader):
            x = x.to(device)
            representation = model(x).cpu().detach().numpy()
            y = y.numpy()
            test_rep_list.append(representation)
            test_y_list.append(y)
    test_rep = np.concatenate(test_rep_list, axis=0)
    test_y = np.concatenate(test_y_list, axis=0)
    y_pred = clf.predict(test_rep)
    prfs=precision_recall_fscore_support(test_y, y_pred, average='weighted', zero_division=0)
    acc = accuracy_score(test_y, y_pred)
    print(acc, prfs)
    return acc, prfs

def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()
    
def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone()-subtrahend[name].data.clone()
    
def reduce_add_average(targets, sources):
    for target in targets:
        for name in target:
            tmp = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()
            target[name].data += tmp
        
def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])

def pairwise_angles(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = flatten(source1)
            s2 = flatten(source2)
            angles[i,j] = torch.sum(s1*s2)/(torch.norm(s1)*torch.norm(s2)+1e-12)
    return angles.numpy()

class FederatedTrainingDevice(object):
    def __init__(self, model_fn, data):
        self.model = model_fn.to(device)
        self.data = data
        self.W = {key : value for key, value in self.model.named_parameters()}

    def evaluate(self, loader=None):
        return eval_op(self.id, self.model, self.train_loader, self.eval_loader if not loader else loader)

  
class Client(FederatedTrainingDevice):
    def __init__(self, model_fn, optimizer_fn, data, idnum, batch_size=128, train_frac=0.8):
        super().__init__(model_fn, data)  
        self.optimizer = optimizer_fn(self.model.parameters())
            
        self.data = data
        n_train = int(len(data)*train_frac)
        n_eval = len(data) - n_train 
        data_train, data_eval = torch.utils.data.random_split(self.data, [n_train, n_eval])

        self.train_loader = GraphDataLoader(
            data_train, batch_size=128, drop_last=True)
        self.eval_loader = GraphDataLoader(
            data_eval, batch_size=128, drop_last=True)
        
        self.id = idnum
        
        self.dW = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        
        self.conv1_dW = {'conv1': torch.zeros_like(self.model.conv1.weight)}
        self.conv2_dW = {'conv2': torch.zeros_like(self.model.conv2.weight)}
        self.conv3_dW = {'conv3': torch.zeros_like(self.model.conv3.weight)}
        
    def synchronize_with_server(self, server):
        copy(target=self.W, source=server.W)
    
    def compute_weight_update(self, epochs=1, loader=None):
        copy(target=self.W_old, source=self.W)
        self.optimizer.param_groups[0]["lr"]*=0.001
        train_stats = train_op(self.model, self.train_loader if not loader else loader, self.optimizer, epochs)
        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)
        return train_stats  

    def reset(self): 
        copy(target=self.W, source=self.W_old)
    
    
class Server(FederatedTrainingDevice):
    def __init__(self, model_fn, data):
        super().__init__(model_fn, data)
        train_sampler = SubsetRandomSampler(torch.arange(len(self.data)))

        self.loader = GraphDataLoader(
            self.data, sampler=train_sampler, batch_size=32, drop_last=False)
        self.model_cache = []
        self.W = {key : value for key, value in self.model.named_parameters()}
        self.id = 1000
    
    def select_clients(self, clients, frac=1.0):
        return random.sample(clients, int(len(clients)*frac)) 
    
    def aggregate_weight_updates(self, clients):
        reduce_add_average(targets=[self.W], sources=[client.dW for client in clients])
        
    def compute_pairwise_similarities(self, clients, layer_wise=True, layer_id=0):
        clients.sort(key=lambda x: x.id)
        if layer_wise == False:
            return pairwise_angles([client.dW for client in clients])
        else:
            if layer_id == 0:
                return pairwise_angles([client.conv1_dW for client in clients])
            elif layer_id == 1:
                return pairwise_angles([client.conv2_dW for client in clients])
            else:
                return pairwise_angles([client.conv3_dW for client in clients])
                
    def cluster_clients(self, S, origin_index):
        clustering = AgglomerativeClustering(affinity="euclidean", linkage="ward").fit(-S)
        origin_index = np.array(origin_index)
        c1 = np.argwhere(clustering.labels_ == 0).flatten() 
        c2 = np.argwhere(clustering.labels_ == 1).flatten()
        c1 = origin_index[c1]
        c2 = origin_index[c2]
        return c1, c2
    
    def aggregate_clusterwise(self, client_clusters):
        for cluster in client_clusters:
            reduce_add_average(targets=[client.W for client in cluster], 
                               sources=[client.dW for client in cluster])

    def compute_max_update_norm(self, cluster):
        return np.max([torch.norm(flatten(client.dW)).item() for client in cluster])

    
    def compute_mean_update_norm(self, cluster):
        return torch.norm(torch.mean(torch.stack([flatten(client.dW) for client in cluster]), 
                                     dim=0)).item()

    def cache_model(self, idcs, params, accuracies):
        self.model_cache += [(idcs, 
                            {name : params[name].data.clone() for name in params}, 
                            [accuracies[i] for i in idcs])]


