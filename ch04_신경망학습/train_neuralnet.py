import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)


# 하이퍼파라미터
iters_num = 10000 # 반복 횟수
train_size = x_train.shape[0] # 60000
batch_size = 100 # 미니배치 크기
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1) # 600

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size) # 0~59999 중 100개 랜덤 선택
    x_batch = x_train[batch_mask] # 100개 랜덤 선택
    t_batch = t_train[batch_mask] # 100개 랜덤 선택

    # 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch) # 기울기 계산
    # grad = network.gradient(x_batch, t_batch) # 성능 개선판! (ch05에서 설명)

    # 매개변수 갱신
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key] # 기울기를 이용해 매개변수 갱신

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch) # 손실함수 값 계산
    train_loss_list.append(loss) # 손실함수 값 기록

    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuarcy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
        

