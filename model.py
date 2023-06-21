from loss_function import cdcca_loss
from layers import WidelyLinearLayer, ComplexActivation, ComplexRelu

import torch
import torch.nn as nn
import numpy as np

class MlpNetwork(nn.Module):
    def __init__(self, layer_sizes, input_size):
        super(MlpNetwork, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        self.complex_activation = ComplexRelu()
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    #ComplexBatchNorm1d(num_features=layer_sizes[l_id], affine=False),
                    WidelyLinearLayer(layer_sizes[l_id], layer_sizes[l_id + 1]),
                ))
            else:
                layers.append(nn.Sequential(
                    WidelyLinearLayer(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    self.complex_activation,
                    #ComplexBatchNorm1d(num_features=layer_sizes[l_id + 1], affine=False),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ComplexDeepCCA(nn.Module):
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size, device=torch.device('cpu')):
        super(ComplexDeepCCA, self).__init__()
        self.model1 = MlpNetwork(layer_sizes1, input_size1).double()
        self.model2 = MlpNetwork(layer_sizes2, input_size2).double()

        self.loss = cdcca_loss(outdim_size, device).loss

        print("\nIMPORTANT: if you are running on GPU, edit this function accordingly, otherwise ignore this message \n")

    def forward(self, x1, x2, device):
        # input1 = torch.randn(x1.size(0),1, dtype=torch.cdouble)
        # input2 = torch.randn(x2.size(0),1, dtype=torch.cdouble)

        # for i in range(0,len(input1)):
        #     input1[i] = torch.complex(x1[i][0][0], x1[i][0][1])
        #     input2[i] = torch.complex(x2[i][0][0], x2[i][0][1])

        # output1 = self.model1(input1.to(device=device))
        # output2 = self.model2(input2.to(device=device))

        # IMPORTANT: if running with GPU uncomment the above lines
        # and comment the following two lines, otherwise leave as it is
        output1 = self.model1(x1)
        output2 = self.model2(x2)

        return output1, output2
