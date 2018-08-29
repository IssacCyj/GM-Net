import matplotlib.pylab as plt
import json
import numpy as np


def plot_cifar10():

    with open("log/boomnet_log_cifar100.json", "r") as f:
        d = json.load(f)

    fig = plt.figure()

    train_erro = 1 - np.array(d["train_loss"])[:, 1]
    test_erro = 1 - np.array(d["test_loss"])[:, 1]
    
    train_loss = np.array(d["train_loss"])[:, 0]
    test_loss = np.array(d["test_loss"])[:, 0]

    ax1 = fig.add_subplot(111)
    plt.plot(100*train_erro, color="lightsalmon", linewidth=1,label='train erro(%)')
    plt.plot(100*test_erro, color="lightskyblue", linewidth=1,label='test erro(%)')
    plt.ylim(0,50)
    ax1.set_ylabel('erro rate(%)')
    

    ax2 = ax1.twinx()
    plt.semilogy(train_loss,'--', color="salmon", linewidth=1,label='train logloss')
    plt.semilogy(test_loss, '--',color="skyblue", linewidth=1,label='test logloss')
    plt.ylim(0.01,10)
    ax2.set_ylabel('logloss')
    lines,labels = ax1.get_legend_handles_labels()
    lines2,labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines+lines2,labels+labels2,loc=0)



    plt.grid()
    plt.show()
    plt.clf()
    plt.close()

if __name__ == '__main__':
    plot_cifar10()
