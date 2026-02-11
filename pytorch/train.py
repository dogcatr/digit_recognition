import matplotlib.pylab as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device=}")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

# 画像データをロード
mnist_data = np.load("mnist.npz")

data = mnist_data['x_train'].astype(np.float32) / 255
data = np.reshape(data, (60000, 784))
t = np.eye(10)[mnist_data['y_train']]
data = torch.from_numpy(data).clone().to(device)
t = torch.from_numpy(t).to(device)

iterations = 10000
batch_size = 100
learning_rate = 0.01
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 正解率のリスト
acc_list = []

for _ in tqdm(range(iterations)):

    # Pytorch用にデータを用意
    batch_mask = np.random.choice(data.shape[0], batch_size)
    data_batch = data[batch_mask]
    t_batch = t[batch_mask]

    # 損失誤差を計算
    pred = model(data_batch)
    loss = loss_fn(pred, t_batch)

    # バックプロパゲーション
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # トレーニングデータに対して正解率を算出(*テストデータに対して行うべきもの)
    batch_mask = np.random.choice(data.shape[0], batch_size)
    data_test_batch = data[batch_mask]
    t_test_batch = t[batch_mask]

    correct = 0
    with torch.no_grad():
        pred = model(data_test_batch)
        correct \
            = (pred.argmax(1) == t_test_batch.argmax(1)).type(torch.float).sum().item()

    acc_list.append(correct / t_test_batch.shape[0])

# 正答率グラフ表示(対トレーニングデータ)
plt.plot(range(iterations), acc_list)
plt.show()
