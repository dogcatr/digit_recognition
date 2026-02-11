
from collections import OrderedDict
import matplotlib.pylab as plt
import numpy as np
from tqdm import tqdm

import functions as fs


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] \
            = fs.Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = fs.Relu()
        self.layers['Affine2'] \
            = fs.Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = fs.SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        return np.sum(y == t) / float(y.shape[0])


# 画像データの取得
mnist_data = np.load("mnist.npz")

data = mnist_data['x_train'].astype(np.float32) / 255
data = np.reshape(data, (60000, 784))
t = np.eye(10)[mnist_data['y_train']]

# ネットワーク作成

network = TwoLayerNet(
    input_size=784,
    hidden_size=50,
    output_size=10,
)

iterations = 10000
batch_size = 100
learning_rate = 0.01

# 正解率のリスト
acc_list = []

# 学習
for _ in tqdm(range(iterations)):
    batch_mask = np.random.choice(data.shape[0], batch_size)
    data_batch = data[batch_mask]
    t_batch = t[batch_mask]
    # print(t_batch)

    grad = network.gradient(data_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    batch_mask = np.random.choice(data.shape[0], batch_size)
    data_batch = data[batch_mask]
    t_batch = t[batch_mask]
    acc_list.append(network.accuracy(data_batch, t_batch))

# トレーニングデータに対する正解率(*テストデータに対して行うべきもの)
plt.plot(range(iterations), acc_list)
plt.show()

# ３つの画像に対して予測
batch_mask = np.random.choice(mnist_data['x_test'].shape[0], 3)
test_data = mnist_data['x_test'][batch_mask]

for i in range(3):
    flatten_data = np.reshape(test_data[i], (1, 784))
    mid_ans = network.predict(flatten_data)
    ans = np.argmax(fs.softmax(mid_ans))
    print(f"この数字は{ans}")
    plt.imshow(test_data[i])
    plt.show()
