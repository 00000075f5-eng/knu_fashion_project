# coding: utf-8
r"""
[2단계] 최종 예시 모델: 기준 모델 + 교재 6장 기법

학습 코드의 뼈대는 baseline_two_layer.py(강의 5.7.4절)와 똑같다.
달라진 곳은 아래 세 군데뿐이다.
  (1) 신경망: TwoLayerNet(은닉 1층) → MultiLayerNetExtend(은닉 3층)
  (2) 갱신 방법: params -= lr * grad (SGD) → optimizer.update(...) (Adam)
  (3) 검증 정확도가 가장 높았던 에폭의 신경망을 따로 저장해 둔다

교재 6장 기법 (※ 6장은 아직 수업 전. 교재 6장을 읽고 이해할 것)
  - He 초깃값         : 6.2절  ReLU에 맞는 가중치 초깃값
  - Adam              : 6.1절  학습률을 매개변수마다 알아서 조절하는 갱신 방법
  - 배치 정규화(BN)    : 6.3절  각 층의 출력 분포를 고르게 맞춰 학습을 안정시킴
  - 드롭아웃          : 6.4절  학습 때 뉴런 일부를 무작위로 꺼서 오버피팅을 줄임
  - 가중치 감소       : 6.4절  큰 가중치에 벌점을 줘서 오버피팅을 줄임

실행: C:\knu_Fashion_Project 폴더에서  python team_project/fashion_example_model.py
결과: team_project/network_TeamX.pkl, final_loss.png, final_accuracy.png
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import pickle
import copy
import numpy as np
import matplotlib.pyplot as plt
from dataset.fashion_mnist import load_fashion_mnist, class_names
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import Adam
from common.util import shuffle_dataset

np.random.seed(0)   # 난수 고정: 실행할 때마다 같은 결과가 나오게 한다
TEAM_NO = '5'       # 우리 조 번호
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 데이터 읽기 ──
(x_train, t_train), (x_test, t_test) = load_fashion_mnist(normalize=True, one_hot_label=True)

# ── 훈련 50,000 + 검증 10,000으로 나누기 (기준 모델과 같은 방법) ──
x_train, t_train = shuffle_dataset(x_train, t_train)
validation_num = 10000
x_val, t_val = x_train[:validation_num], t_train[:validation_num]
x_train, t_train = x_train[validation_num:], t_train[validation_num:]

# ── 신경망 만들기 ──
# 784 → 256 → 128 → 64 → 10  (은닉층 3개, Affine 층 4개 → "6층 이하" 조건 만족)
network = MultiLayerNetExtend(input_size=784, hidden_size_list=[256, 128, 64], output_size=10,
                              activation='relu',
                              weight_init_std='he',          # He 초깃값 (6.2절)
                              weight_decay_lambda=5e-7,      # 가중치 감소 (6.4절)
                              use_dropout=True, dropout_ration=0.2,   # 드롭아웃 (6.4절)
                              use_batchnorm=True)            # 배치 정규화 (6.3절)
optimizer = Adam(lr=0.0025)                                   # Adam (6.1절)

# ── 하이퍼파라미터 ──
max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = max(train_size / batch_size, 1)   # 500번 반복이 1에폭
iters_num = int(max_epochs * iter_per_epoch)       # 20에폭 = 10,000번 반복

train_loss_list = []
train_acc_list = []
val_acc_list = []
best_val_acc = 0
best_network = None


def clear_cache(net):
    """계층들이 순전파 때 기억해 둔 중간값(입력 x, mask 등)을 지운다.
    5장에서 배운 것처럼 각 계층은 역전파를 위해 순전파 입력을 저장해 둔다.
    지우지 않고 pickle로 저장하면 그 데이터까지 들어가 파일이 수십 MB가 된다.
    (클래스 코드는 고치지 않고, 객체에 남은 값만 비운다)"""
    for layer in list(net.layers.values()) + [net.last_layer]:
        for name in ('x', 'original_x_shape', 'mask', 'out', 'dW', 'db', 'xc', 'xn',
                     'std', 'batch_size', 'input_shape', 'dgamma', 'dbeta', 'y', 't', 'loss'):
            if hasattr(layer, name):
                setattr(layer, name, None)


for i in range(iters_num):
    # 1단계: 미니배치 추출
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 2단계: 기울기 계산 (오차역전파법)
    grads = network.gradient(x_batch, t_batch)

    # 3단계: 매개변수 갱신 (Adam)
    optimizer.update(network.params, grads)

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1에폭마다 훈련/검증 정확도 기록
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        val_acc = network.accuracy(x_val, t_val)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        epoch = len(val_acc_list) - 1
        print("epoch", epoch, "| train acc, val acc |", round(train_acc, 4), ",", round(val_acc, 4))

        # 검증 정확도가 지금까지 중 가장 높으면 이 순간의 신경망을 복사해 둔다
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            clear_cache(network)
            best_network = copy.deepcopy(network)

print("\n가장 좋은 val acc:", round(best_val_acc, 4), "(epoch", best_epoch, ")")

# ── 학습이 끝난 뒤, 시험 데이터로 딱 한 번 평가 ──
test_acc = best_network.accuracy(x_test, t_test)
print("[최종 모델] test acc:", round(test_acc, 4))

# 옷 종류별 정확도: 어떤 옷을 많이 틀리는지 → 발표의 "한계 분석"에 활용
y = np.argmax(best_network.predict(x_test, train_flg=False), axis=1)
t = np.argmax(t_test, axis=1)
print("\n옷 종류별 test 정확도")
for k in range(10):
    print(" ", k, class_names[k].ljust(12), round(np.mean(y[t == k] == k), 3))

# ── Competition 제출 파일 저장 (3.6.2절에서 배운 pickle) ──
clear_cache(best_network)
pkl_path = os.path.join(SAVE_DIR, 'network_Team' + str(TEAM_NO) + '.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump(best_network, f)
print("\n저장:", pkl_path)

# ── 그래프: 손실 추이 ──
plt.figure()
plt.plot(np.arange(len(train_loss_list)), train_loss_list)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.title("final model")
plt.savefig(os.path.join(SAVE_DIR, 'final_loss.png'), dpi=100)

# ── 그래프: 정확도 추이 ──
plt.figure()
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train acc')
plt.plot(x, val_acc_list, marker='s', label='val acc', linestyle='--')
plt.axvline(best_epoch, color='gray', linestyle=':', label='best epoch')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.title("final model")
plt.savefig(os.path.join(SAVE_DIR, 'final_accuracy.png'), dpi=100)
plt.show()
