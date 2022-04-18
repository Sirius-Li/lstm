import numpy as np

class Sigmod(object):
    def forward(self, x):
        if x.all() >= 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出
            return 1.0 / (1 + np.exp(-x))
        else:
            return np.exp(x) / (1 + np.exp(x))

class Tanh(object):
    def forward(self, x):
        return (np.exp(x) - np.exp(-1 * x)) / (np.exp(x) + np.exp(-1 * x))

class LossFunction(object):
    def CheckLoss(self, Predict, Y):
        # 平方和
        # print(np.concatenate((Predict, Y), axis=1))
        # print(Predict.shape)
        # print(Y.shape)
        count = Predict.size
        sub = Predict - Y
        LOSS = sub * sub
        loss = np.sum(LOSS) / count
        correct = LOSS < 1e-3
        return LOSS.reshape(-1), correct.reshape(-1)

class LstmCell(object):
    def __init__(self, input_size, hidden_size, bias=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

    #??????????
    def split_params(self, matrix, size):
        return matrix[0:1*size, :], \
               matrix[1*size:2*size, :], \
               matrix[2*size:3*size, :], \
               matrix[3*size:4*size, :]

    def get_params(self, W, U, b=None):
        self.wf, self.wi, self.wg, self.wo = self.split_params(W, self.hidden_size)
        self.uf, self.ui, self.ug, self.uo = self.split_params(U, self.input_size)
        self.bf, self.bi, self.bg, self.bo = self.split_params((b if self.bias else np.zeros((4, self.hidden_size))), 1)

    def get_gate(self, h, x, W, U, b, activator):
        if self.bias:
            # TODO
            z = np.dot(h, W) + np.dot(x, U) + b
        else:
            z = np.dot(h, W) + np.dot(x, U)
        a = activator.forward(z)
        return a

    def forward(self, x, h_p, c_p, W, U, b=None):
        #W (h, h)*4  U(i, H)*4
        self.get_params(W, U, b)
        self.x = x

        # use g instead of c_temp
        self.f = self.get_gate(h_p, x, self.wf, self.uf, self.bf, Sigmod())
        self.i = self.get_gate(h_p, x, self.wi, self.ui, self.bi, Sigmod())
        self.g = self.get_gate(h_p, x, self.wg, self.ug, self.bg, Tanh())
        self.o = self.get_gate(h_p, x, self.wo, self.uo, self.bo, Sigmod())

        self.c = np.multiply(self.f, c_p) + np.multiply(self.i, self.g)
        self.h = np.multiply(self.o, Tanh().forward(self.c))


    def backward(self, h_p, c_p, in_grad):
        tanh = lambda x: Tanh().forward(x)

        self.dc  = in_grad * self.o * (1 - tanh(self.c) * tanh(self.c))
        self.dzo = in_grad * tanh(self.c) * self.o * (1 - self.o)
        self.dzg = self.dc * self.i * (1 - self.g * self.g)
        self.dzi = self.dc * self.g * self.i * (1 - self.i)
        self.dzf = self.dc * c_p * self.f * (1 - self.f)

        self.dwo = np.dot(h_p.T, self.dzo)
        self.dwg = np.dot(h_p.T, self.dzg)
        self.dwi = np.dot(h_p.T, self.dzi)
        self.dwf = np.dot(h_p.T, self.dzf)

        self.duo = np.dot(self.x.T, self.dzo)
        self.dug = np.dot(self.x.T, self.dzg)
        self.dui = np.dot(self.x.T, self.dzi)
        self.duf = np.dot(self.x.T, self.dzf)

        if self.bias:
            self.dbo = np.sum(self.dzo, axis=0, keepdims=True)
            self.dbg = np.sum(self.dzg, axis=0, keepdims=True)
            self.dbi = np.sum(self.dzi, axis=0, keepdims=True)
            self.dbf = np.sum(self.dzf, axis=0, keepdims=True)

        # h_p previous time
        self.dh = np.dot(self.dzf, self.wf.T) + np.dot(self.dzi, self.wi.T) + \
                  np.dot(self.dzg, self.wg.T) + np.dot(self.dzo, self.wo.T)
        # x_p previous layer
        self.dx = np.dot(self.dzf, self.uf.T) + np.dot(self.dzi, self.ui.T) + \
                  np.dot(self.dzg, self.ug.T) + np.dot(self.dzo, self.uo.T)

    def merge_params(self):
        self.dW = np.concatenate((self.dwf, self.dwi, self.dwg, self.dwo), axis=0)
        self.dU = np.concatenate((self.duf, self.dui, self.dug, self.duo), axis=0)
        if self.bias:
            self.db = np.concatenate((self.dbf, self.dbi, self.dbg, self.dbo), axis=0)


class LinearCell(object):
    def __init__(self, input_size, output_size, activator=None, bias=True):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        self.bias = bias

    def forward(self, x, V, b=None):
        self.x = x
        self.batch_size = self.x.shape[0]
        self.V = V
        self.b = b if self.bias else np.zeros((self.output_size))
        self.z = np.dot(x, V) + self.b
        if self.activator:
            self.a = self.activator.forward(self.z)
        else:
            self.a = self.z

    def backward(self, in_grad):
        self.dz = in_grad
        self.dV = np.dot(self.x.T, self.dz)
        if self.bias:
            self.db = np.sum(self.dz, axis=0, keepdims=True)
        self.dx = np.dot(self.dz, self.V.T)