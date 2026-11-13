# coding: utf-8
# 6장 학습 관련 기술들(1) 실습문제 4번
# optimizer_compare_mnist.py를 Fashion-MNIST 데이터로 학습하도록 수정
# 준비: 프로젝트의 dataset/fashion/ 에 있는 4개 압축 파일과 dataset/mnist_reader.py 를 사용
# 실행: C:\knu_Fashion_Project 폴더에서  python team_project/experiments/compare_optimizer.py
import sys, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')   # C:\knu_Fashion_Project
sys.path.append(ROOT)  # common, dataset 폴더를 가져올 수 있도록 프로젝트 최상위 폴더를 추가

import numpy as np
import matplotlib.pyplot as plt
from dataset import mnist_reader
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *


# 0. Fashion-MNIST 데이터 읽기==========
data_dir = os.path.join(ROOT, 'dataset', 'fashion')
x_train, t_train = mnist_reader.load_mnist(data_dir, kind='train')
x_test, t_test = mnist_reader.load_mnist(data_dir, kind='t10k')

# 픽셀 값을 0~1로 정규화 (mnist_reader는 0~255 정수 그대로 돌려줌)
x_train = x_train / 255.0
x_test = x_test / 255.0

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000     # 그래프를 보면서 적절한 값으로 조절


# 1. 실험용 설정==========
optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()
optimizers['Adam'] = Adam()

networks = {}
train_loss = {}
for key in optimizers.keys():
    networks[key] = MultiLayerNet(
        input_size=784, hidden_size_list=[100, 100, 100],
        output_size=10)
    train_loss[key] = []


# 2. 훈련 시작==========
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)

        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))


# 3. 그래프 그리기==========
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x = np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
# 그래프를 png 파일로 저장 (이 파일과 같은 폴더에 compare_optimizer.png)
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compare_optimizer.png'), dpi=100)
plt.show()
