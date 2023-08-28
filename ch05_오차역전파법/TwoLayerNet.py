import sys, os
sys.path.append(os.pardir)
import numpy as np

from common.layers import *
from collections import OrderedDict

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) # 정규분포로 초기화
        self.params['b1'] = np.zeros(hidden_size) # 편향은 0으로 초기화
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) # 정규분포로 초기화
        self.params['b2'] = np.zeros(output_size) # 편향은 0으로 초기화

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss() # 마지막 계층은 SoftmaxWithLoss 계층