

# 나쁜 구현 예
def numerical_diff_bad(f, x):   ## 수치미분
    h = 10e-50
    return (f(x + h) - f(x)) / h


# 좋은 구현 예
def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)  ## 중앙차분