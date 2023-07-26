import sys, os
sys.path.append("C:/Users/user/Desktop/Deep Learning/DeepLearning_from_scratch")
import numpy as np
from dataset.mnist import load_mnist
import pickle


# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 소프트맥스 함수
def softmax(a):
    c = np.max(a) # overflow 방지
    exp_a = np.exp(a-c) # 오버플로 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("ch03/sample_weight.pkl", "rb") as f:
        network = pickle.load(f)

        return network
    
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3'] # 가중치 매개변수
    b1, b2, b3 = network['b1'], network['b2'], network['b3'] # 편향 매개변수

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1) # 활성화 함수로 시그모이드 함수 사용
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3) # 활성화 함수로 소프트맥스 함수 사용

    return y


x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)): # 10000번 반복
    y = predict(network, x[i])
    p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻음
    if p == t[i]:
        accuracy_cnt += 1
print("Accuracy: " + str(float(accuracy_cnt) / len(x)))