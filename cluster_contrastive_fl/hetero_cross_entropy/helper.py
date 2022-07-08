import numpy as np
import matplotlib.pyplot as plt

class ExperimentLogger:
    def log(self, values):
        for k, v in values.items():
            if k not in self.__dict__:
                self.__dict__[k] = [v]
            else:
                self.__dict__[k] += [v]



def display_train_stats(cfl_stats, eps_1, eps_2, communication_rounds):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    acc_mean = np.mean(cfl_stats.acc_clients, axis=1)
    acc_std = np.std(cfl_stats.acc_clients, axis=1)
    plt.plot(cfl_stats.rounds, acc_mean, color="C0")
    plt.plot(cfl_stats.rounds, acc_std, color="C1")
    
    if "split" in cfl_stats.__dict__:
        for s in cfl_stats.split:
            plt.axvline(x=s, linestyle="-", color="k", label=r"Split")
    
    
    plt.text(x=communication_rounds, y=1, ha="right", va="top", 
             s="Clusters: {}".format([x for x in cfl_stats.clusters[-1]])) # this is a log which accumulates history records
    
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")
    
    plt.xlim(0, communication_rounds)
    plt.ylim(0,1)
    
    plt.subplot(1,2,2)
    
    plt.plot(cfl_stats.rounds, cfl_stats.mean_norm, color="C1", label=r"$\|\sum_i\Delta W_i \|$")
    plt.plot(cfl_stats.rounds, cfl_stats.max_norm, color="C2", label=r"$\max_i\|\Delta W_i \|$")
    
    plt.axhline(y=eps_1, linestyle="--", color="k", label=r"$\varepsilon_1$")
    plt.axhline(y=eps_2, linestyle=":", color="k", label=r"$\varepsilon_2$")

    if "split" in cfl_stats.__dict__:
        for s in cfl_stats.split:
            plt.axvline(x=s, linestyle="-", color="k", label=r"Split")

    plt.xlabel("Communication Rounds")
    plt.legend()
    
    plt.xlim(0, communication_rounds)
    
    plt.show()