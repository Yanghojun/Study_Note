import torch
import torch.optim

# Data definition
x_train = torch.Tensor([[73, 80, 75], [93, 88, 93], [89, 91, 80], [96, 98, 100], [73, 66, 70]])
y_train = torch.Tensor([[152], [185], [180], [196], [142]])

# W variable definition
# W = torch.zeros_like(x_train, requires_grad=True)
W = torch.zeros((3,1), requires_grad=True)              # 처음에 이부분 이해 안갔었음. x_train이 5x3 이니까 이 W도 5x3 이여야 할줄 알았는데 아니였어
                                                        # 왜냐하면 이 W의 3x1은 x_train 5개의 데이터가 모두 동시에 쓰는 값들이기 때문이야.
                                                        # 5개의 데이터를 하나의 신경망에 넣는거지. 5개의 데이터를 5개의 신경망에 넣는게 아니란것을 생각하면 이해가 갈꺼야. 과거의 내가 미래의 나한테..
b = torch.zeros(1, requires_grad=True)

# Optimizer definition
optimizer = torch.optim.SGD([W, b], lr=1e-5)

# Decision of epoch and start training
num_epoch = 20
for epoch in range(num_epoch):

    # Inference
    hypothesis = x_train.matmul(W) + b

    # cost
    cost = torch.mean((hypothesis - y_train)**2)

    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(epoch, num_epoch, hypothesis.squeeze().detach(), cost.item()))

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


# 추가적으로 nn.Module에 대해서 설명하자면
    # nn.Module을 상속해서 모델을 생성하고
    # nn.Linear(3,1) 등을 활용해서 모델을 구성할 수 있다.
    # Hypothesis 계산은 forward()에서 해주고!
    # Gradient 계산은 Pytorch가 알아서 backward()로 해준다
    # F.mse_loss(prediction, y_train) 등을 통해서 cost function을 쉽게 할 수 있게 해준다. 버그도 없고, 다른 cost Function과의 교체도 매우 용이하다.
    # 쉽게 말해서 훨씬 간단하게 모델을 구성할 수 있게 해주는 것이다. nn.Module이


