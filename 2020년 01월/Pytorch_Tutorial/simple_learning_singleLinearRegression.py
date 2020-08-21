#   Linear regression : 선형회귀
import torch
import torch.optim

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

# 모델 초기화
W = torch.zeros(1, requires_grad=True)

# Learning rate 설정
lr = 0.1

optimizer = torch.optim.SGD([W], lr=0.15)

nb_epochs = 10
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x_train*W

    # Cost gradient 계산
    cost = torch.mean((hypothesis - y_train)**2)

    # gradient = torch.sum((W*x_train - y_train) * x_train)     실습코드는 이걸로 해주셨는데
    # gradient = torch.mean((W * x_train -  y_train) * x_train)    # 이렇게 평균을 내서 하는게 맞을듯.

    print('Epoch {:4d}/{} W: {:.3f}, Cost: {:6f}' .format(epoch, nb_epochs, W.item(), cost.item()))

    # Cost gradient로 H(x) 개선
    # W -= lr * gradient

    # torch.optim 사용해서 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()