import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000 # 반복 횟수
train_size = x_train.shape[0] # 훈련 데이터의 개수
batch_size = 100 # 미니배치 크기
learning_rate = 0.1 # 학습률

train_loss_list = [] # 훈련 손실 함수의 값이 담길 리스트
train_acc_list = [] # 훈련 데이터에 대한 정확도가 담길 리스트
test_acc_list = [] # 시험 데이터에 대한 정확도가 담길 리스트

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask] # 미니배치
    t_batch = t_train[batch_mask] # 미니배치

    # 오차 역전파법으로 기울기를 구한다.
    grad = network.gradient(x_batch, t_batch)

    # 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss= network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
        