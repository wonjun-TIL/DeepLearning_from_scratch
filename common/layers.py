import numpy as np
from functions import *

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0) # x가 0보다 작거나 같으면 True, 아니면 False
        out = x.copy() # x를 복사
        out[self.mask] = 0 # x가 0보다 작거나 같으면 0으로 바꿈

        return out
    
    def backward(self, dout):
        dout[self.mask] = 0 # x가 0보다 작거나 같으면 0으로 바꿈
        dx = dout # dout을 복사

        return dx
    

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1+np.exp(-x)) # 시그모이드 함수
        self.out = out # out에 저장
        
        return out
    
    def backward(self, dout):
        dx = dout* (1.0 - self.out) * self.out # 시그모이드의 미분

        return dx
    

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout) # x.T: x의 전치행렬
        self.db = np.sum(dout, axis=0)

        return dx
    
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실
        self.y = None # softmax의 출력
        self.t = None # 정답 레이블 (원-핫 벡터)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx