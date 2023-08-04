import numpy as np


# 정답 레이블이 숫자 레이블일 때 
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size) ## 1차원이면 2차원으로 바꿔줌
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size)] + 1e-7)) / batch_size

