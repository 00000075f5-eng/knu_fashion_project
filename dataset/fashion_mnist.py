# coding: utf-8
"""
Fashion-MNIST 데이터셋 읽기

수업에서 쓴 dataset/mnist.py의 load_mnist()와 똑같은 모양으로 쓰도록 만든 파일이다.
    (x_train, t_train), (x_test, t_test) = load_fashion_mnist(normalize=True, one_hot_label=True)

실제 파일 읽기는 팀플 안내문에 나온 공식 로더(dataset/mnist_reader.py)의 load_mnist()를 사용한다.
데이터 파일은 dataset/fashion/ 폴더의 .gz 파일 4개.
"""
import os
import numpy as np
from dataset.mnist_reader import load_mnist as _read_fashion   # 안내문의 공식 로더

dataset_dir = os.path.dirname(os.path.abspath(__file__))
fashion_dir = os.path.join(dataset_dir, 'fashion')

# 레이블 번호 → 옷 이름 (0~9)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def _change_one_hot_label(X):
    """정수 레이블을 원-핫 인코딩으로 바꾼다 (mnist.py와 같은 방식)"""
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T


def load_fashion_mnist(normalize=True, flatten=True, one_hot_label=False):
    """Fashion-MNIST 데이터셋 읽기

    Parameters
    ----------
    normalize : 이미지의 픽셀 값을 0.0~1.0 사이의 값으로 정규화할지 정한다.
    flatten : 입력 이미지를 1차원 배열(784)로 만들지를 정한다.
    one_hot_label : True면 레이블을 원-핫 배열로 돌려준다.

    Returns
    -------
    (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
    """
    x_train, t_train = _read_fashion(fashion_dir, kind='train')   # 60,000장
    x_test, t_test = _read_fashion(fashion_dir, kind='t10k')      # 10,000장

    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

    if one_hot_label:
        t_train = _change_one_hot_label(t_train)
        t_test = _change_one_hot_label(t_test)

    if not flatten:
        x_train = x_train.reshape(-1, 1, 28, 28)
        x_test = x_test.reshape(-1, 1, 28, 28)

    return (x_train, t_train), (x_test, t_test)
