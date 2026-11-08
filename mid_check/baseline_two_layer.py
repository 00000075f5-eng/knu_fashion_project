# coding: utf-8
r"""
[1단계] 기준 모델(베이스라인): 수업에서 배운 2층 신경망 그대로

- 강의 5.7.4절 "오차역전파법을 사용한 학습 구현하기" 코드를 Fashion-MNIST에 맞게 옮긴 것
- 신경망 설정도 수업과 같다: 입력 784, 은닉 50, 출력 10, 학습률 0.1, 배치 100, SGD
- 바뀐 점 두 가지
  (1) MNIST 대신 Fashion-MNIST를 읽는다.
  (2) 학습 중에는 test 대신 validation(검증 데이터)으로 정확도를 본다.
      → 팀플 규칙: "학습시 학습데이터와 검증데이터만 활용, 테스트 데이터는 성능평가 때만 사용"

이 모델의 정확도가 이후 모든 실험의 비교 기준이 된다.

실행: C:\knu_Neural_Net 폴더에서  python mid_check/baseline_two_layer.py
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.fashion_mnist import load_fashion_mnist
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset

np.random.seed(0)   # 난수 고정: 실행할 때마다 같은 결과가 나오게 한다

# ── 데이터 읽기 ──
(x_train, t_train), (x_test, t_test) = load_fashion_mnist(normalize=True, one_hot_label=True)

# ── 훈련 데이터 60,000장을 훈련 50,000 + 검증 10,000으로 나누기 ──
# (교재 6.5.1절 방식: 먼저 섞은 뒤 앞쪽을 검증 데이터로 쓴다)
x_train, t_train = shuffle_dataset(x_train, t_train)
validation_num = 10000
x_val, t_val = x_train[:validation_num], t_train[:validation_num]
x_train, t_train = x_train[validation_num:], t_train[validation_num:]

# ── 신경망 만들기 ──
# 은닉층이 1개인 MultiLayerNet = 수업의 TwoLayerNet과 같은 구조
#   Affine1 → Relu1 → Affine2 → SoftmaxWithLoss
# (수업과 똑같이 가중치 표준편차 0.01로 초기화)
network = MultiLayerNet(input_size=784, hidden_size_list=[50], output_size=10,
                        activation='relu', weight_init_std=0.01)

# ── 하이퍼파라미터 (수업 5.7.4절과 같은 값) ──
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
val_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)   # 50,000 / 100 = 500번 반복이 1에폭

for i in range(iters_num):
    # 1단계: 미니배치 추출
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 2단계: 기울기 계산 (오차역전파법)
    grad = network.gradient(x_batch, t_batch)

    # 3단계: 매개변수 갱신 (확률적 경사하강법, SGD)
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1에폭마다 훈련/검증 정확도 기록
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        val_acc = network.accuracy(x_val, t_val)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        print("epoch", len(val_acc_list) - 1, "| train acc, val acc |",
              round(train_acc, 4), ",", round(val_acc, 4))

# ── 학습이 끝난 뒤, 시험 데이터로 딱 한 번 평가 ──
test_acc = network.accuracy(x_test, t_test)
print("\n[기준 모델] 최종 val acc:", round(val_acc_list[-1], 4), "| test acc:", round(test_acc, 4))

# ── 그래프: 손실 추이 (그림 4-11 방식) ──
plt.figure()
plt.plot(np.arange(len(train_loss_list)), train_loss_list)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.title("baseline (TwoLayerNet, SGD)")
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline_loss.png'), dpi=100)

# ── 그래프: 정확도 추이 (그림 4-12 방식) ──
plt.figure()
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train acc')
plt.plot(x, val_acc_list, marker='s', label='val acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.title("baseline (TwoLayerNet, SGD)")
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline_accuracy.png'), dpi=100)
plt.show()
