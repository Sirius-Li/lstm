import math

import matplotlib.pyplot as plt
import numpy as np
from data import DataSet
from lstm import *

import torch
# torch.lstm()

class Net(object):
    def __init__(self, dr, input_size, hidden_size, output_size, times=4, bias=True):
        self.dr = dr
        self.loss_fun = LossFunction()
        self.times = times
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bias = bias
        self.lstmcell = []
        self.linearcell = []
        for i in range (self.times):
            self.lstmcell.append(LstmCell(input_size, hidden_size, bias))
            self.linearcell.append(LinearCell(hidden_size, output_size, activator=None, bias=bias))


    def forward(self, X):
        hp = np.zeros((self.batch_size, self.hidden_size))
        cp = np.zeros((self.batch_size, self.hidden_size))
        for i in range(self.times):
            x = X[:, i].reshape(-1, 1)
            self.lstmcell[i].forward(x, hp, cp, self.W, self.U, self.bh)
            hp = self.lstmcell[i].h
            cp = self.lstmcell[i].c
            self.linearcell[i].forward(hp, self.V, self.b)

    def backward(self, Y):
        hp = []
        cp = []
        t1 = self.times - 1
        dz = self.linearcell[t1].a - Y[:, t1:t1+1]
        self.linearcell[t1].backward(dz)
        dh = self.linearcell[t1].dx
        hp = self.lstmcell[t1-1].h
        cp = self.lstmcell[t1-1].c
        self.lstmcell[t1].backward(hp, cp, dh)

        dh = []
        for i in range(t1-1, 0, -1):
            dz = self.linearcell[i].a - Y[:, i:i+1]
            self.linearcell[i].backward(dz)
            dh = self.linearcell[i].dx + self.lstmcell[i+1].dh
            hp = self.lstmcell[i - 1].h
            cp = self.lstmcell[i - 1].c
            self.lstmcell[i].backward(hp, cp, dh)

        dz = self.linearcell[0].a - Y[:, 0:1]
        self.linearcell[0].backward(dz)
        dh = self.linearcell[0].dx + self.lstmcell[1].dh
        hp = np.zeros((self.batch_size, self.hidden_size))
        cp = np.zeros((self.batch_size, self.hidden_size))
        self.lstmcell[0].backward(hp, cp, dh)

    def init_params_uniform(self, shape):
        p = []
        std = 1.0 / math.sqrt(self.hidden_size)
        p = np.random.uniform(-std, std, shape)
        return p

    def train(self, batch_size, checkpoint=0.1, max_epoch=100, eta=0.1):
        self.batch_size = batch_size
        # Try different initialize method
        # self.U = np.random.random((4 * self.input_size, self.hidden_size))
        # self.W = np.random.random((4 * self.hidden_size, self.hidden_size))
        self.U = self.init_params_uniform((4 * self.input_size, self.hidden_size))
        self.W = self.init_params_uniform((4 * self.hidden_size, self.hidden_size))
        self.V = np.random.random((self.hidden_size, self.output_size))
        self.bh = np.zeros((4, self.hidden_size))
        self.b = np.zeros((self.output_size))

        max_iteration = math.floor(self.dr.num_train / batch_size)   #向下取整
        checkpoint_iteration = (int)(math.floor(max_iteration * checkpoint))

        LOSS = []
        for epoch in range(max_epoch):
            # TODO
            dr.Shuffle()
            for iteration in range(max_iteration):
                # TODO getData
                batch_x, batch_y = self.dr.GetBatchTrainSample(batch_size, iteration)
                # print(batch_x.shape, batch_y.shape)

                self.forward(batch_x)
                self.backward(batch_y)
                #update
                for i in range(self.times):
                    self.lstmcell[i].merge_params()

                    # print(self.lstmcell[i].dU, self.lstmcell[i].dW)

                    self.U = self.U - self.lstmcell[i].dU * eta / self.batch_size
                    self.W = self.W - self.lstmcell[i].dW * eta / self.batch_size
                    self.V = self.V - self.linearcell[i].dV * eta / self.batch_size
                    if self.bias:
                        self.bh = self.bh - self.lstmcell[i].db * eta / self.batch_size
                        self.b = self.b - self.linearcell[i].db * eta / self.batch_size
                # loss
                total_iteration = epoch * max_iteration + iteration
                if (total_iteration + 1) % checkpoint_iteration == 0:
                    # TODO
                    X, Y = self.dr.GetValidationSet()
                    loss, acc = self.check_loss(X, Y)
                    LOSS.append(loss)
                    # self.loss_trace.Add(epoch, total_iteration, None, None, loss, acc, None)
                    print(epoch, total_iteration)
                    print(str.format("loss={0:6f}, acc={1:6f}", loss, acc))
            plt.plot(LOSS)
            plt.show()

            if acc == 1.0:
                print("ok")
                break

    def check_loss(self, X, Y):
        # # self.forward(X)
        # result = np.zeros(Y.shape)
        # # print(Y.shape)
        # for i in range(X.shape[0]):
        #     self.forward(X[i].reshape(1, -1))
        #     for j in range(Y.shape[1]):
        #         result[i, j] = self.linearcell[j].a


        t_batch_size = self.batch_size
        self.batch_size = X.shape[0]
        self.forward(X)
        self.batch_size = t_batch_size
        result = self.linearcell[0].a
        for i in range(Y.shape[1] - 1):
            np.append(result, self.linearcell[i+1].a, axis=1)

        loss_list = np.zeros(Y.shape)
        acc_list = np.zeros(Y.shape)
        for i in range(self.times):
            loss_list[:, i], acc_list[:, i] = self.loss_fun.CheckLoss(self.linearcell[i].a, Y[:, i:(i + 1)])
            # loss_list[:, i], acc_list[:, i] = self.loss_fun.CheckLoss(result[:, i:(i+1)], Y[:, i:(i + 1)])
            # print(np.concatenate((self.linearcell[i].a, Y[:, i:(i + 1)]), axis=1))

        loss = np.mean(loss_list)
        acc = np.mean(acc_list)
        return loss, acc

    def test(self):
        print("testing...")
        # TODO
        X, Y = [], []
        count = X.shape[0]
        loss, acc, result = self.check_loss(X, Y)
        print(str.format(f"loss={0:6f}, acc={1:6f}", loss, acc))
        r = np.random.randint(0, count, 10)
        for i in range(10):
            idx = r[i]
            x1 = X[idx, :, 0]
            x2 = X[idx, :, 1]
            print("sin:", reverse(x1))
            print("cos:", reverse(x2))
            x1_dec = int("".join(map(str, reverse(x1))), 2)
            x2_dec = int("".join(map(str, reverse(x2))), 2)
            print("{0} - {1} = {2}".format(x1_dec, x2_dec, (x1_dec - x2_dec)))
            print("====================")


def reverse(self, a):
    l = a.tolist()
    l.reverse()
    return l

if __name__ == '__main__':
    #TODO
    dr = DataSet()
    dr.InitTrainDate(num_feature=64)
    input_size = 1
    hidden_size = 64
    output_size = 1

    n = Net(dr, input_size, hidden_size, output_size, times=dr.num_feature, bias=True)
    n.train(batch_size=8, checkpoint=0.1, eta=0.001)

    # n.test()