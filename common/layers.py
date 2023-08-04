
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