"""
An example for the model class
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphs.weights_initializer import weights_init
import numpy as np
import utils.graphUtils.graphML as gml
import utils.graphUtils.graphTools
from torchsummaryX import summary

class DecentralPlannerNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.S = None
        self.numAgents = self.config.num_agents
        # inW = self.config.map_w
        # inH = self.config.map_h

        inW = 11
        inH = 11

        convW = [inW]
        convH = [inH]
        numAction = 5

        hidden_dim = 64
        gamma = 0.98
        epsilon = 0.01
        target_update = 10

        use_vgg = False

        # ------------------ DCP v0 - only for testing
        # # CNN - v3
        # numChannel = [3, 8, 16, 32]
        # numStride = [1, 1, 1]
        #
        # # -- compressMLP
        # dimCompressMLP = 2
        # numCompressFeatures = [2 ** 8, 64]
        # dimNodeSignals = [64]  # only for testing
        # nGraphFilterTaps = [2]
        # dimActionMLP = 2
        # numActionFeatures = [32, numAction]

        # ------------------ DCP v1 - Previous valided Network
        # # CNN - v3
        # numChannel = [3, 8, 16, 32, 32]
        # numStride = [1, 1, 2, 2]
        #
        # # -- compressMLP
        # dimCompressMLP = 2
        # numCompressFeatures = [2**9, 128]
        # numCompressFeatures = [2**8, 128]

        # ------------------ DCP v1.1 - modified
        # # CNN - v3
        # numChannel = [3, 32, 32, 64, 64, 128]
        # numStride = [1, 1, 1, 2, 2]
        #
        # # -- compressMLP
        # dimCompressMLP = 2
        # numCompressFeatures = [2**9,  2 ** 7]
        #
        # dimCompressMLP = 1
        # numCompressFeatures = [2 ** 7] #, 2 ** 7]

        # ------------------ DCP v1.2  -  with maxpool
        # numChannel = [3] +[32, 32, 64, 64, 128]
        # numStride = [1, 1, 1, 2, 2]
        #
        # dimCompressMLP = 1
        # numCompressFeatures = [2**7]
        #
        # nMaxPoolFilterTaps = 2
        # numMaxPoolStride = 2
        # dimNodeSignals = [2 ** 7]

        # ------------------ DCP v1.3  -  with maxpool + non stride in CNN
        # numChannel = [3] +[32, 32, 64, 64, 128]
        # numStride = [1, 1, 1, 1, 1]
        #
        # dimCompressMLP = 1
        # numCompressFeatures = [2**7]
        #
        # nMaxPoolFilterTaps = 2
        # numMaxPoolStride = 2
        # dimNodeSignals = [2 ** 7]

        # ------------------ DCP v1.4  -  with maxpool + non stride in CNN - less feature
        numChannel = [3] + [32, 32, 64, 64, 128]
        numStride = [1, 1, 1, 1, 1]

        dimCompressMLP = 1
        numCompressFeatures = [2 ** 7]

        nMaxPoolFilterTaps = 2
        numMaxPoolStride = 2
        # # 1 layer origin
        dimNodeSignals = [2 ** 7]

        # # 2 layer - upsampling
        # dimNodeSignals = [256, 2 ** 7]

        # # 2 layer - down sampling
        # dimNodeSignals = [64, 2 ** 7]
        #
        # # 2 layer - down sampling -v2
        # dimNodeSignals = [64, 32]
        #
        # ------------------ DCP v2 - 1121
        # numChannel = [3] + [64, 64, 128, 128]
        # numStride = [1, 1, 2, 1]
        #
        #
        # dimCompressMLP = 3
        # numCompressFeatures = [2 ** 12, 2 ** 9, 128]

        # ------------------ DCP v3 - vgg
        # numChannel = [3] + [64, 128, 256, 256, 512, 512, 512, 512]
        # numStride = [1, 1, 2, 1, 1, 2, 1, 1]
        #
        # dimCompressMLP = 3
        # numCompressFeatures = [2 ** 12, 2 ** 12, 128]

        # ------------------ DCP v4 - vgg with max pool & dropout
        # use_vgg = True
        # cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

        ## ------------------ GCN -------------------- ##
        # dimNodeSignals = [2 ** 7]
        # nGraphFilterTaps = [self.config.nGraphFilterTaps,self.config.nGraphFilterTaps] # [2]
        nGraphFilterTaps = [self.config.nGraphFilterTaps]
        # --- actionMLP
        dimActionMLP = 1
        numActionFeatures = [numAction]


        #####################################################################
        #                                                                   #
        #                CNN to extract feature                             #
        #                                                                   #
        #####################################################################

        convl = []
        numConv = len(numChannel) - 1
        nFilterTaps = [3] * numConv
        nPaddingSzie = [1] * numConv
        for l in range(numConv):
            convl.append(nn.Conv2d(in_channels=numChannel[l], out_channels=numChannel[l + 1],
                                   kernel_size=nFilterTaps[l], stride=numStride[l], padding=nPaddingSzie[l],
                                   bias=True))
            convl.append(nn.BatchNorm2d(num_features=numChannel[l + 1]))
            convl.append(nn.ReLU(inplace=True))

            W_tmp = int((convW[l] - nFilterTaps[l] + 2 * nPaddingSzie[l]) / numStride[l]) + 1
            H_tmp = int((convH[l] - nFilterTaps[l] + 2 * nPaddingSzie[l]) / numStride[l]) + 1
            # Adding maxpooling
            if l % 2 == 0:
                convl.append(nn.MaxPool2d(kernel_size=2))
                W_tmp = int((W_tmp - nMaxPoolFilterTaps) / numMaxPoolStride) + 1
                H_tmp = int((H_tmp - nMaxPoolFilterTaps) / numMaxPoolStride) + 1
                # http://cs231n.github.io/convolutional-networks/
            convW.append(W_tmp)
            convH.append(H_tmp)

        self.ConvLayers = nn.Sequential(*convl)

        numFeatureMap = numChannel[-1] * convW[-1] * convH[-1]

        #####################################################################
        #                                                                   #
        #                MLP-feature compression                            #
        #                                                                   #
        #####################################################################

        numCompressFeatures = [numFeatureMap] + numCompressFeatures

        compressmlp = []
        for l in range(dimCompressMLP):
            compressmlp.append(
                nn.Linear(in_features=numCompressFeatures[l], out_features=numCompressFeatures[l + 1], bias=True))
            compressmlp.append(nn.ReLU(inplace=True))

        self.compressMLP = nn.Sequential(*compressmlp)

        self.numFeatures2Share = numCompressFeatures[-1]

        #####################################################################
        #                                                                   #
        #                    graph neural network                           #
        #                                                                   #
        #####################################################################

        self.L = len(nGraphFilterTaps)  # Number of graph filtering layers
        self.F = [numCompressFeatures[-1]] + dimNodeSignals  # Features [128,128]
        # self.F = [numFeatureMap] + dimNodeSignals  # Features
        self.K = nGraphFilterTaps  # nFilterTaps # Filter taps
        self.E = 1  # Number of edge features
        self.bias = True

        gfl = []  # Graph Filtering Layers
        for l in range(self.L):
            # \\ Graph filtering stage:
            gfl.append(gml.GraphFilterBatch(self.F[l], self.F[l + 1], self.K[l], self.E, self.bias))
            # There is a 2*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.

            # \\ Nonlinearity
            gfl.append(nn.ReLU(inplace=True))

        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl)  # Graph Filtering Layers

        #####################################################################
        #                                                                   #
        #                    MLP --- map to actions                         #
        #                                                                   #
        #####################################################################

        # numActionFeatures = [self.F[-1]] + numActionFeatures
        # self.gamma = gamma  # 折扣因子
        # self.epsilon = epsilon  # epsilon-贪婪策略
        # self.target_update = target_update  # 目标网络更新频率
        # self.count = 0  # 计数器,记录更新次数
        # self.q_net = Qnet(self.F[-1], hidden_dim, numAction)  # Q网络
        # # 目标网络
        # self.target_q_net = Qnet(self.F[-1], hidden_dim, numAction)
        # actionsfc = []
        # for l in range(dimActionMLP):
        #     if l < (dimActionMLP - 1):
        #         actionsfc.append(
        #             nn.Linear(in_features=numActionFeatures[l], out_features=numActionFeatures[l + 1], bias=True))
        #         actionsfc.append(nn.ReLU(inplace=True))
        #     else:
        #         actionsfc.append(
        #             nn.Linear(in_features=numActionFeatures[l], out_features=numActionFeatures[l + 1], bias=True))
        #
        # self.actionsMLP = nn.Sequential(*actionsfc)
        dqn = []
        dqn.append(nn.Linear(in_features=128, out_features=64, bias=True))
        dqn.append(nn.ReLU(inplace=False))
        dqn.append(nn.Linear(in_features=64, out_features=5, bias=True))
        self.dqn = nn.Sequential(*dqn)
        self.apply(weights_init)

    def make_layers(self, cfg, batch_norm=False):
        layers = []

        input_channel = 3
        for l in cfg:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue

            layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

            if batch_norm:
                layers += [nn.BatchNorm2d(l)]

            layers += [nn.ReLU(inplace=True)]
            input_channel = l

        return nn.Sequential(*layers)


    def addGSO(self, S):

        # We add the GSO on real time, this GSO also depends on time and has
        # shape either B x N x N or B x E x N x N
        if self.E == 1:  # It is B x T x N x N
            assert len(S.shape) == 3
            self.S = S.unsqueeze(1)  # B x E x N x N
        else:
            assert len(S.shape) == 4
            assert S.shape[1] == self.E
            self.S = S

    def forward(self, inputTensor,next):
        # inputTensor (64,10,3,11,11)
        B = inputTensor.shape[0] # batch size 64

        # B x G x N
        extractFeatureMap = torch.zeros(B, self.numFeatures2Share, self.numAgents).to(self.config.device) # (64,128,10)
        for id_agent in range(self.numAgents):
            input_currentAgent = inputTensor[:, id_agent] # (64,3,11,11)
            featureMap = self.ConvLayers(input_currentAgent) # (64,128,1,1)
            featureMapFlatten = featureMap.view(featureMap.size(0), -1) # (64,128)
            # extractFeatureMap[:, :, id_agent] = featureMapFlatten
            compressfeature = self.compressMLP(featureMapFlatten) # (64,128)
            extractFeatureMap[:, :, id_agent] = compressfeature # B x F x N (64,128,10)

        # DCP
        for l in range(self.L):
            # \\ Graph filtering stage:
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            self.GFL[2 * l].addGSO(self.S) # add GSO for GraphFilter

        # B x F x N - > B x G x N,
        sharedFeature = self.GFL(extractFeatureMap) # (64,128,10)

        action_predict = []
        if next == False:
            for id_agent in range(self.numAgents):
                # DCP_nonGCN
                # sharedFeature_currentAgent = extractFeatureMap[:, :, id_agent]
                # DCP
                # torch.index_select(sharedFeature_currentAgent, 3, id_agent)
                sharedFeature_currentAgent = sharedFeature[:, :, id_agent] # (64,128)
                # print("sharedFeature_currentAgent.requires_grad: {}\n".format(sharedFeature_currentAgent.requires_grad))
                # print("sharedFeature_currentAgent.grad_fn: {}\n".format(sharedFeature_currentAgent.grad_fn))

                sharedFeatureFlatten = sharedFeature_currentAgent.view(sharedFeature_currentAgent.size(0), -1)
                # print('TYPE:', type(sharedFeatureFlatten))
                action_currentAgents = self.dqn(sharedFeatureFlatten) # (64,5)
                action_predict.append(action_currentAgents) # N x 5

            return self.dqn, action_predict # (10[64,5])

        else:
            return sharedFeature # (64,128,10)

class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action