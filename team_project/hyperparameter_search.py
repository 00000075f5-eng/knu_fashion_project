# coding: utf-8
r"""
[⑤ 하이퍼파라미터 최적화] Randomized Grid Search로 학습률과 가중치 감소 계수 찾기

강의 6.5.2~6.5.3절, 교재 ch06/hyperparameter_optimization.py 방식을 그대로 따른다.
  0단계: 하이퍼파라미터 값의 범위를 설정
  1단계: 그 범위에서 값을 무작위로 추출 (학습률과 가중치 감소 계수를 한 쌍으로 뽑는다)
  2단계: 그 값으로 학습하고, 검증 데이터로 정확도 평가 (빠르게 하려고 에폭은 작게)
  3단계: 반복한 뒤 정확도가 높은 값들을 보고 범위를 좁혀 다시 탐색

- 1차 탐색: 강의와 같은 범위 (가중치 감소 1e-8~1e-4, 학습률 1e-6~1e-2), 30회
- 2차 탐색: 1차 상위 5개를 모두 포함하는 범위로 좁혀서 20회
- 신경망 구조와 나머지 기법(He, 배치 정규화, 드롭아웃, Adam)은 최종 모델과 같다.
- 시간을 줄이려고 훈련 데이터 10,000장, 5에폭만 학습한다. (강의: "에폭은 작게 설정해도 괜찮다")
- 시험 데이터는 전혀 쓰지 않는다.

실행: C:\knu_Fashion_Project 폴더에서  python team_project/hyperparameter_search.py   (약 5~10분)
결과: 화면에 Best-1~5 출력, hyperparameter_search.png 저장
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.fashion_mnist import load_fashion_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.util import shuffle_dataset
from common.trainer import Trainer

np.random.seed(0)

# ── 데이터: 최종 모델과 똑같이 나눈 뒤, 훈련 데이터는 10,000장만 사용 ──
(x_train, t_train), (x_test, t_test) = load_fashion_mnist(normalize=True)
x_train, t_train = shuffle_dataset(x_train, t_train)
validation_num = 10000
x_val, t_val = x_train[:validation_num], t_train[:validation_num]
x_train, t_train = x_train[validation_num:], t_train[validation_num:]
x_train, t_train = x_train[:10000], t_train[:10000]     # 결과를 빠르게 얻기 위해 훈련 데이터를 줄임


def __train(lr, weight_decay, epocs=5):
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[256, 128, 64], output_size=10,
                                  activation='relu', weight_init_std='he',
                                  weight_decay_lambda=weight_decay,
                                  use_dropout=True, dropout_ration=0.2, use_batchnorm=True)
    # 교재처럼 Trainer를 쓰고, 평가 데이터 자리에 검증 데이터를 넣는다
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epocs, mini_batch_size=100,
                      optimizer='adam', optimizer_param={'lr': lr}, verbose=False)
    trainer.train()
    return trainer.test_acc_list, trainer.train_acc_list   # test_acc_list에는 검증 정확도가 들어 있다


def search(trial_num, wd_range, lr_range, results):
    """wd_range, lr_range는 10의 지수 범위. 예: (-8, -4) → 1e-8 ~ 1e-4"""
    for _ in range(trial_num):
        # 탐색할 하이퍼파라미터를 한 쌍으로 무작위 추출
        weight_decay = 10 ** np.random.uniform(wd_range[0], wd_range[1])
        lr = 10 ** np.random.uniform(lr_range[0], lr_range[1])

        val_acc_list, train_acc_list = __train(lr, weight_decay)
        print("val acc:" + str(round(val_acc_list[-1], 4)) + " | lr:" + format(lr, '.2e') +
              ", weight decay:" + format(weight_decay, '.2e'))
        results.append((val_acc_list[-1], lr, weight_decay, val_acc_list, train_acc_list))


# ── 1차 탐색: 강의와 같은 넓은 범위 ──
print("=========== 1차 탐색: weight decay 1e-8~1e-4, lr 1e-6~1e-2 ===========")
results1 = []
search(30, (-8, -4), (-6, -2), results1)

# ── 범위 좁히기: 상위 5개를 모두 포함하는 구간 (강의 6.5.3절) ──
top5 = sorted(results1, key=lambda r: r[0], reverse=True)[:5]
lr_logs = [np.log10(r[1]) for r in top5]
wd_logs = [np.log10(r[2]) for r in top5]
lr_range2 = (min(lr_logs), max(lr_logs))
wd_range2 = (min(wd_logs), max(wd_logs))
print("\n1차 상위 5개를 포함하도록 범위를 좁힘")
print("  lr:", format(10 ** lr_range2[0], '.1e'), "~", format(10 ** lr_range2[1], '.1e'))
print("  weight decay:", format(10 ** wd_range2[0], '.1e'), "~", format(10 ** wd_range2[1], '.1e'))

# ── 2차 탐색: 좁힌 범위 ──
print("\n=========== 2차 탐색 ===========")
results2 = []
search(20, wd_range2, lr_range2, results2)

# ── 결과 ──
print("\n=========== Hyper-Parameter Optimization Result ===========")
all_results = sorted(results1 + results2, key=lambda r: r[0], reverse=True)
for i, r in enumerate(all_results[:5]):
    print("Best-" + str(i + 1) + "(val acc:" + str(round(r[0], 4)) + ") | lr:" + format(r[1], '.2e') +
          ", weight decay:" + format(r[2], '.2e'))

# ── 그래프: 1차·2차 탐색에서 뽑은 값들의 위치와 정확도 ──
plt.figure(figsize=(7, 5))
for res, marker, label in ((results1, 'o', '1st search'), (results2, '^', '2nd search')):
    lrs = [r[1] for r in res]
    wds = [r[2] for r in res]
    accs = [r[0] for r in res]
    sc = plt.scatter(lrs, wds, c=accs, cmap='viridis', vmin=0.5, vmax=0.9, marker=marker, label=label, edgecolors='k')
best = all_results[0]
plt.scatter([best[1]], [best[2]], s=250, facecolors='none', edgecolors='red', linewidths=2, label='best')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("learning rate")
plt.ylabel("weight decay")
plt.colorbar(sc, label='val acc')
plt.legend(loc='upper center')
plt.title("Randomized search (5 epochs, 10,000 train)")
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hyperparameter_search.png'), dpi=100)
plt.show()
