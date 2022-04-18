import numpy as np
# from keras.datasets import imdb
import matplotlib.pyplot as plt

class DataSet(object):
    def __init__(self):
        self.num_train = 0  # num of training examples
        self.num_test = 0  # num of test examples
        self.num_validation = 0  # num of validation examples
        self.num_feature = 0  # num of features
        # self.num_category = 0  # num of categories
        self.XTrain = None  # training feature set
        self.YTrain = None  # training label set
        self.XTest = None  # test feature set
        self.YTest = None  # test label set
        self.XDev = None  # valiadtion feature set
        self.YDev = None  # validation feature set

    def InitTrainDate(self, num_feature = 128, validation_rate = 0.1):
        # (self.train_data, self.train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


        T = 10000
        time = np.arange(1, T+1, dtype=np.float)
        train_sin = np.sin(np.pi * time / 8)
        train_cos = np.cos(np.pi * time / 8)

        # plt.plot(train_sin, train_cos, 'sin', 'cos')
        # plt.show()

        feature = np.zeros((T - num_feature, num_feature))
        labels = np.zeros((T - num_feature, num_feature))
        for i in range(num_feature):
            feature[:, i] = train_sin[i: T - num_feature + i]
            labels[:, i] = train_cos[i: T - num_feature + i]
        # labels = train_cos[num_feature:].reshape((-1, 1))

        self.XTrain = feature
        self.YTrain = labels
        self.num_feature = num_feature
        self.num_train = self.XTrain.shape[0]

        self.num_validation = (int)(self.num_train * validation_rate)
        self.num_train = self.num_train - self.num_validation
        self.XDev = self.XTrain[0:self.num_validation, :]
        self.YDev = self.YTrain[0:self.num_validation, :]
        self.XTrain = self.XTrain[self.num_validation:, :]
        self.YTrain = self.YTrain[self.num_validation:, :]

    def GetBatchTrainSample(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        batch_X = self.XTrain[start:end, :]
        batch_Y = self.YTrain[start:end, :]
        return batch_X, batch_Y

    def GetValidationSet(self):
        return self.XDev, self.YDev

    def Shuffle(self):
        seed = np.random.randint(0, 100)
        np.random.seed(seed)
        self.XTrain = np.random.permutation(self.XTrain)
        np.random.seed(seed)
        self.YTrain = np.random.permutation(self.YTrain)


