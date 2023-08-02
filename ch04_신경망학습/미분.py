

# 나쁜 구현 예
def numerical_diff(f, x):   ## 수치미분
    h = 10e-50
    return (f(x + h) - f(x)) / h