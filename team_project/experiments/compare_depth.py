# coding: utf-8
r"""
[비교 실험 ②] 은닉층 깊이 × 너비 비교 (그리드 탐색)

13주 팀플 항목 "모델 결정: 층 수, 층별 노드 수 등 검증데이터 사용"에 해당하는 실험이다.

- 바꾸는 것 : 은닉층 수 {1, 2, 3, 4} × 층마다 뉴런 수 {64, 128, 256} = 12가지 조합
- 고정하는 것: Adam(lr=0.001), He 초깃값, 배치 정규화, 드롭아웃 0.2,
               가중치 감소 5e-7, 배치 100, 10에폭
- 시드 2개로 반복해 평균과 표준편차를 낸다
- 비교 기준  : 에폭별 검증 정확도 중 "가장 좋았던 에폭"의 값 (마지막 에폭 값은 출렁임이 커서 쓰지 않는다)
- 검증 데이터로만 비교하고, 시험 데이터는 쓰지 않는다

실행: C:\knu_Fashion_Project 폴더에서  python team_project/experiments/compare_depth.py
결과: 화면에 결과 표, compare_depth_width.png 저장 (약 20~40분)
"""
import sys, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
sys.path.append(ROOT)  # C:\knu_Fashion_Project 를 경로에 추가 (common, dataset을 가져오기 위해)
import numpy as np
import matplotlib.pyplot as plt
from dataset.fashion_mnist import load_fashion_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import Adam
from common.util import shuffle_dataset

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 데이터: 최종 모델과 똑같이 훈련 50,000 / 검증 10,000으로 나눈다 ──
np.random.seed(0)
(x_train, t_train), (x_test, t_test) = load_fashion_mnist(normalize=True, one_hot_label=True)
x_train, t_train = shuffle_dataset(x_train, t_train)
validation_num = 10000
x_val, t_val = x_train[:validation_num], t_train[:validation_num]
x_train, t_train = x_train[validation_num:], t_train[validation_num:]

# ── 실험 설정 ──
depths = [1, 2, 3, 4]
widths = [64, 128, 256]
combos = [(d, w) for d in depths for w in widths]   # 12가지
seeds = [0, 1]
max_epochs = 10
batch_size = 100
train_size = x_train.shape[0]
iter_per_epoch = int(max(train_size / batch_size, 1))   # 500
iters_num = max_epochs * iter_per_epoch                  # 5,000

train_acc = {c: [] for c in combos}   # 시드별 에폭마다 훈련 정확도
val_acc = {c: [] for c in combos}     # 시드별 에폭마다 검증 정확도

for seed in seeds:
    print("=========== seed", seed, "===========")
    for (d, w) in combos:
        np.random.seed(seed)
        network = MultiLayerNetExtend(input_size=784, hidden_size_list=[w] * d, output_size=10,
                                      activation='relu', weight_init_std='he',
                                      weight_decay_lambda=5e-7,
                                      use_dropout=True, dropout_ration=0.2, use_batchnorm=True)
        optimizer = Adam(lr=0.001)
        tr_list, va_list = [], []

        for i in range(iters_num):
            batch_mask = np.random.choice(train_size, batch_size)          # 1단계: 미니배치
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]
            grads = network.gradient(x_batch, t_batch)                     # 2단계: 기울기 (오차역전파)
            optimizer.update(network.params, grads)                        # 3단계: 갱신 (Adam)

            if (i + 1) % iter_per_epoch == 0:   # 1에폭마다 기록 (훈련 정확도는 시간 절약을 위해 1만 장만)
                tr_list.append(network.accuracy(x_train[:10000], t_train[:10000]))
                va_list.append(network.accuracy(x_val, t_val))

        train_acc[(d, w)].append(tr_list)
        val_acc[(d, w)].append(va_list)
        print("은닉층", d, "개 x 너비", w, "| best val", round(max(va_list), 4),
              "(에폭", int(np.argmax(va_list)) + 1, ") | 마지막 val", round(va_list[-1], 4), flush=True)

# ── 조합별 요약: 시드마다 "가장 좋았던 에폭"의 검증 정확도를 구한 뒤 시드 평균 ──
best_mean, best_std, best_epoch, gap = {}, {}, {}, {}
for c in combos:
    va = np.array(val_acc[c])            # (시드, 에폭)
    tr = np.array(train_acc[c])
    best_idx = va.argmax(axis=1)         # 시드별 최고 에폭
    best_va = va.max(axis=1)
    best_tr = tr[np.arange(len(seeds)), best_idx]
    best_mean[c] = best_va.mean()
    best_std[c] = best_va.std()
    best_epoch[c] = best_idx.mean() + 1
    gap[c] = (best_tr - best_va).mean()

# ── 결과 표 (검증 정확도 높은 순) ──
print("\n=========== 깊이 x 너비 결과 (가장 좋았던 에폭의 val, 시드 " + str(len(seeds)) + "개 평균) ===========")
print("은닉층 | 너비 | best val 평균 ± 표준편차 | 평균 best 에폭 | train−val 차이")
for c in sorted(combos, key=lambda k: -best_mean[k]):
    print(str(c[0]).rjust(5), "|", str(c[1]).rjust(4), "|",
          format(best_mean[c] * 100, '.2f') + "% ± " + format(best_std[c] * 100, '.2f') + "%p", "|",
          format(best_epoch[c], '.1f'), "|",
          format(gap[c] * 100, '.2f') + "%p")
best_c = max(combos, key=lambda k: best_mean[k])
print("\n검증 기준 최고 조합: 은닉층", best_c[0], "개 x 너비", best_c[1],
      "(" + format(best_mean[best_c] * 100, '.2f') + "%)")

# ── 그래프: 왼쪽 깊이별 best val(너비별 선), 오른쪽 조합×깊이 히트맵 ──
plt.figure(figsize=(12, 4.5))

plt.subplot(1, 2, 1)
for w in widths:
    m = [best_mean[(d, w)] for d in depths]
    s = [best_std[(d, w)] for d in depths]
    plt.errorbar(depths, m, yerr=s, marker='o', capsize=3, label='width ' + str(w))
plt.xticks(depths)
plt.xlabel("number of hidden layers")
plt.ylabel("best validation accuracy (mean of seeds)")
plt.title("Best-epoch val accuracy by depth")
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
grid = np.array([[best_mean[(d, w)] for w in widths] for d in depths]) * 100
plt.imshow(grid, cmap='viridis', aspect='auto')
plt.colorbar(label='best val acc (%)')
plt.xticks(range(len(widths)), widths)
plt.yticks(range(len(depths)), depths)
plt.xlabel("width (neurons per layer)")
plt.ylabel("hidden layers")
for i in range(len(depths)):
    for j in range(len(widths)):
        plt.text(j, i, format(grid[i, j], '.2f'), ha='center', va='center', color='w', fontsize=9)
plt.title("Best val accuracy (%)")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'compare_depth_width.png'), dpi=100)
plt.show()
