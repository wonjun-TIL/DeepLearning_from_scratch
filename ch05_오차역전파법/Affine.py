import numpy as np

def softmax(x):
    if x.ndim == 2:
        x = x.T # 행과 열 바꾸기
        x = x - np.max(x, axis=0) # 각 데이터의 최댓값을 빼준다 (오버플로 대책)
        y = np.exp(x) / np.sum(np.exp(x), axis=0) # softmax
        return y.T # 다시 행과 열 바꾸기
    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1: # 데이터가 하나인 경우
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    if t.size == y.size: # 정답 레이블이 원-핫 벡터인 경우
        t = t.argmax(axis=1) # 정답 레이블의 인덱스만 추출
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size



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