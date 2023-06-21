
from utils import load_data, split_data, plot_data
from linear_cca import linear_cca
from model import ComplexDeepCCA

import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler

import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import math
import os
import time
import logging

torch.set_default_tensor_type(torch.DoubleTensor)

class Trainer():
    def __init__(self, model, linear_cca, outdim_size, epoch_num, batch_size, learning_rate, reg_par, device=torch.device('cpu')):
        self.model = nn.DataParallel(model) #nn.parallel.DistributedDataParallel(model)
        self.model.to(device)
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.loss = model.loss
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate, weight_decay=reg_par)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = device

        self.linear_cca = linear_cca
        self.outdim_size = outdim_size

        #create a logger
        self.logger = logging.getLogger('mylogger')
        level = logging.INFO #logging.DEBUG
        self.logger.setLevel(level)

        if (self.logger.hasHandlers()):
            self.logger.handlers.clear()

        handler = logging.FileHandler('DCCA.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.propagate = False

        ch = logging.StreamHandler()
        ch.setLevel(level)
        self.logger.addHandler(ch)

        self.logger.info(self.model)
        self.logger.info("Number of model parameters: {}".format(len(list(self.model.parameters()))))
        self.logger.info("Model Parameters:")
        for i in range(len(list(self.model.parameters()))):
            self.logger.info(list(self.model.parameters())[i].size())

        self.logger.info(self.optimizer)

    def train(self, x1, x2, vx1=None, vx2=None, tx1=None, tx2=None, checkpoint='checkpoint.model'):
        """
        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]
        """
        x1.to(self.device, dtype = torch.cfloat)
        x2.to(self.device, dtype = torch.cfloat)

        data_size = x1.size(0)

        if vx1 is not None and vx2 is not None:
            best_val_loss = 0
            vx1.to(self.device)
            vx2.to(self.device)
        if tx1 is not None and tx2 is not None:
            tx1.to(self.device)
            tx2.to(self.device)

        train_losses = []
        train_loss_history = []
        val_loss_history = []
        for epoch in range(self.epoch_num):
            epoch_start_time = time.time()
            self.model.train()
            batch_idxs = list(BatchSampler(SequentialSampler(range(data_size)), batch_size=self.batch_size, drop_last=False)) #RandomSampler #SequentialSampler
            #with torch.autograd.profiler.profile() as prof:
            for batch_idx in batch_idxs:
                self.optimizer.zero_grad()
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                self.optimizer.zero_grad()
                out1, out2 = self.model(batch_x1, batch_x2, self.device)
                loss = self.loss(out1, out2)
                train_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            train_loss = np.mean(train_losses)
            #print(prof.key_averages().table(sort_by="self_cpu_time_total"))

            info_string = "Epoch {:d}/{:d} - time: {:.2f} - training_loss: {:.4f}"
            if vx1 is not None and vx2 is not None:
                with torch.no_grad():
                    self.model.eval()
                    val_loss = self.transform(vx1, vx2)
                    info_string += " - val_loss: {:.4f}".format(-val_loss)
                    if val_loss < best_val_loss:
                        self.logger.info(
                            "Epoch {:d}: val_loss improved from {:.4f} to {:.4f}, saving model to {}".format(epoch + 1, -best_val_loss, -val_loss, checkpoint))
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(), checkpoint)
                    else:
                        self.logger.info("Epoch {:d}: val_loss did not improve from {:.4f}".format(epoch + 1, -best_val_loss))
            else:
                torch.save(self.model.state_dict(), checkpoint)
            epoch_time = time.time() - epoch_start_time
            self.logger.info(info_string.format(epoch + 1, self.epoch_num, epoch_time, -train_loss))

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)

        # train_linear_cca
        if self.linear_cca is not None:
            _, outputs = self._get_outputs(x1, x2)
            self.linear_cca.fit(outputs[0], outputs[1], self.outdim_size)

        checkpoint_ = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint_)

        if vx1 is not None and vx2 is not None:
            loss = self.transform(vx1, vx2)
            self.logger.info("loss on validation data: {:.4f}".format(-loss))

        if tx1 is not None and tx2 is not None:
            loss = self.transform(tx1, tx2)
            self.logger.info('loss on test data: {:.4f}'.format(-loss))

        history = [-1 * np.array(train_loss_history), -1 * np.array(val_loss_history)]

        plt.figure(figsize=(10,4))
        epochs = np.arange(1, epoch_num+1, step=1)
        epochs1 = np.arange(0, epoch_num+9, step=10)
        plt.plot(epochs, history[0], color = 'tab:blue', label='Training')
        plt.plot(epochs, history[1], color = 'tab:orange', label='Validation')
        plt.xticks(epochs1)
        plt.xlim(0, 52)
        plt.title('Model Correlation')
        plt.ylabel('Correlation')
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid()
        plt.show()

        return history

    def transform(self, x1, x2, use_linear_cca=False):
        with torch.no_grad():
            losses, outputs = self._get_outputs(x1, x2)

            if use_linear_cca:
                print("Linear CCA started!")
                outputs = self.linear_cca.transform(outputs[0], outputs[1])
                return losses, outputs
                # return np.mean(losses), outputs
            else:
                return np.mean(losses)

    def _get_outputs(self, x1, x2):
        with torch.no_grad():
            self.model.eval()
            data_size = x1.size(0)
            batch_idxs = list(BatchSampler(SequentialSampler(range(data_size)), batch_size=self.batch_size, drop_last=False))
            losses = []
            outputs1 = []
            outputs2 = []
            for batch_idx in batch_idxs:
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                out1, out2 = self.model(batch_x1, batch_x2, self.device)
                outputs1.append(out1)
                outputs2.append(out2)
                loss = self.loss(out1, out2)
                losses.append(loss.item())
        outputs = [torch.cat(outputs1, dim=0).cpu().numpy(),
                   torch.cat(outputs2, dim=0).cpu().numpy()]
        return losses, outputs   

if __name__ == '__main__':
    ############
    # Parameters Section

    # select 'device' depending on personal device information
    # if GPU is availabe, then uncomment the next three lines
    # device = torch.device('cuda')
    # print("Using", torch.cuda.device_count(), "GPUs")
    # print("GPU Name:", torch.cuda.get_device_name())

    # otherwise use CPU:
    device = torch.device('cpu')

    # the path to save the final learned features
    save_to = './new_features.gz'

    # the size of the new space learned by the model (number of the new features)
    outdim_size = 3

    # size of the input for view 1 and view 2
    input_shape1 = input_shape2 = 1

    # number of layers with nodes in each one
    # 5 components
    # layer_sizes1 = [256, 512, 1024, 1024, 1024, 512, outdim_size]
    # layer_sizes2 = [256, 512, 1024, 1024, 1024, 512, outdim_size]

    # 3 components
    layer_sizes1 = [128, 256, 512, 1024, outdim_size]
    layer_sizes2 = [128, 256, 512, 1024, outdim_size]

    # 2 components
    # layer_sizes1 = [128, 256, 512, outdim_size]
    # layer_sizes2 = [128, 256, 512, outdim_size]

    # the parameters for training the network
    learning_rate = 1e-3
    epoch_num = 1
    batch_size = 50

    # the regularization parameter of the network - necessary to avoid the gradient exploding/vanishing
    reg_par = 1e-5

    # if a linear CCA should get applied on the learned features
    apply_linear_cca = True

    ############

    # Load data and plot it 

    # tick1 = 'AAPL'
    # tick2 = 'PFE'
    tick1 = '^GSPC'
    tick2 = '^DJI'
    start = '1992-01-01'
    end = '2023-01-01'

    #plot_data(tick1, tick2, start, end)

    view1 = load_data(tick1, start, end)
    view2 = load_data(tick2, start, end)

    # Split data
    train_ratio=0.7
    val_ratio=0.15
    train1, val1, test1 = split_data(view1, train_ratio=train_ratio, val_ratio=val_ratio)
    train2, val2, test2 = split_data(view2, train_ratio=train_ratio, val_ratio=val_ratio)

    # Set up CDCCA model and train 
    # Building, training, and producing the new features by CDCCA
    model = ComplexDeepCCA(layer_sizes1, layer_sizes2, input_shape1, input_shape2, outdim_size, device=device).double()
    l_cca = None
    if apply_linear_cca:
        l_cca = linear_cca()
    trainer = Trainer(model, l_cca, outdim_size, epoch_num, batch_size, learning_rate, reg_par, device=device)

    history = trainer.train(train1, train2, val1, val2, test1, test2)

    # transform the views
    train_loss1, train_outputs = trainer.transform(train1, train2, apply_linear_cca)
    val_loss1, val_outputs = trainer.transform(val1, val2, apply_linear_cca)
    test_loss1, test_outputs = trainer.transform(test1, test2, apply_linear_cca)

    print("Training data canonical correlation", -np.mean(train_loss1))
    print("Validation data canonical correlation", -np.mean(val_loss1))
    print("Testing data canonical correlation", -np.mean(test_loss1))

    output1 = [train_outputs[0], val_outputs[0], test_outputs[0]]
    output2 = [train_outputs[1], val_outputs[1], test_outputs[1]]

    dataset = ["Training", "Validation", "Testing"]

    # create figures for Real parts vs each each other 
    fig = plt.figure(constrained_layout=True, figsize=(25, 10))
    subfigs = fig.subfigures(nrows=3, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f'{dataset[row]}', fontweight="bold")

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=outdim_size)
        for col, ax in enumerate(axs):
            ax.scatter(output1[row][:, col].real, output2[row][:, col].real)

            corr = np.corrcoef(output1[row][:, col].real, output2[row][:, col].real)[0, 1]
            ax.set_title("Component {}".format(col + 1))
            ax.set_xlabel('View 1')
            ax.set_ylabel('View 2')
            ax.grid()
    plt.show()

    # create figures for Imaginary parts vs each each other 
    fig = plt.figure(constrained_layout=True, figsize=(25, 10))
    subfigs = fig.subfigures(nrows=3, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f'{dataset[row]}', fontweight="bold")

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=outdim_size)
        for col, ax in enumerate(axs):
            ax.scatter(output1[row][:, col].imag, output2[row][:, col].imag)

            ax.set_title("Component {}".format(col + 1))
            ax.set_xlabel('View 1')
            ax.set_ylabel('View 2')
            ax.grid()
    plt.show()

    
    d = torch.load('checkpoint.model')
    trainer.model.load_state_dict(d)
    trainer.model.parameters()
    