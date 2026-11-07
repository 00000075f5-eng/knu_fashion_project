# coding: utf-8
r"""
[3단계] Competition 리허설

교수님은 우리 코드를 실행하지 않고, 제출한 pkl 파일만 불러와 정확도를 구한다.
여기서는 그 상황을 흉내 낸다: pkl만 불러와서 test 데이터 200장으로 정확도를 구한다.
(실제 대회는 교수님이 주는 test_data.pkl과 Competition_Code_NeuralNet.py로 진행)

실행: C:\knu_Fashion_Project 폴더에서  python team_project/check_competition.py
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import pickle
import numpy as np
from dataset.fashion_mnist import load_fashion_mnist

TEAM_NO = '5'   # fashion_example_model.py와 같은 번호
pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'network_Team' + str(TEAM_NO) + '.pkl')

# ── 1. pkl 파일 불러오기 (3.6.2절의 sample_weight.pkl 읽기와 같은 방법) ──
with open(pkl_path, 'rb') as f:
    network = pickle.load(f)
print("불러온 객체:", type(network).__name__)

# ── 2. 시험 데이터 읽기 (학습 때와 똑같이 normalize=True) ──
(x_train, t_train), (x_test, t_test) = load_fashion_mnist(normalize=True, one_hot_label=True)

# ── 3. 시험 데이터 중 200장을 무작위로 뽑아 정확도 구하기, 5번 반복 ──
np.random.seed(0)
for trial in range(5):
    mask = np.random.choice(x_test.shape[0], 200, replace=False)
    acc = network.accuracy(x_test[mask], t_test[mask])
    print("200장 시험", trial + 1, "회차 | accuracy:", acc, "(" + str(int(round(acc * 200))) + "개 정답)")

# ── 4. 주의: 정규화를 빠뜨리면? (학습 때와 전처리가 다르면 점수가 떨어진다) ──
acc_wrong = network.accuracy(x_test[mask] * 255.0, t_test[mask])
print("\n[주의] 0~255 원본 픽셀을 그대로 넣었을 때 accuracy:", acc_wrong)
