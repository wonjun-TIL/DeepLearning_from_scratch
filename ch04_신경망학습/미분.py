

# 나쁜 구현 예
def numerical_diff_bad(f, x):   ## 수치미분
    h = 10e-50
    return (f(x + h) - f(x)) / h


# 좋은 구현 예
def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)  ## 중앙차분


def function_1(x):
    return 0.01*x**2 + 0.1*x


# 함수 그리기
import numpy as np
import matplotlib.pylab as plt

x = np.arange(0.0, 20.0, 0.1)  # 0에서 20까지 0.1 간격의 배열 x를 만든다(20은 미포함).
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()


# 편미분
def function_2(x):
    return x[0]**2 + x[1]**2
    # 또는 return np.sum(x**2)


# 기울기
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2&h) ## 중앙차분
        x[idx] = tmp_val # 값 복원

    return grad