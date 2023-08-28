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

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    
    # x: 입력 데이터, t: 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)
    
    def accuracy(self, x, t):
        y = self.predcit(x)
        y = np.argmax(y, axis=1) # 최대값의 인덱스를 반환
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y==t) / float(x.shape[0])

        return accuracy
    
    # x: 입력 데이터, t: 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W : self.loss(x,t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads
    
    def gradient(self, x, t):
        # 순전파
        self.loss(x, t)

        # 역전파
        dout = 1
        dout = self.lastLayer.backward(dout)

        # 역전파를 통해 기울기를 구함
        layers = list(self.layers.values())
        layers.reverse() # 역전파는 역순으로 진행해야함
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'] = self.layers['Affine1'].dw
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dw
        grads['b2'] = self.layers['Affine2'].db

        return grads
    
    