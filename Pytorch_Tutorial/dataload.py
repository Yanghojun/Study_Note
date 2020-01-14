import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MultivariateLinearRegressionModel(nn.Module):     # nn.Module 클래스를 상속하겠다!
    def __init__(self):
        super().__init__()      # super는 부모클래스를 가리키고 이 의미는 부모클래스의 __init__()을 호출한다는 의미이다. 이걸 호출해서 필요한 속성들을 초기화 하는 것이다.
                                # 예를들어 이걸 안해주면 nn.Module의 __init()__에서 정의된 변수등을 사용할 수 없다.

        self.linear = nn.Linear(3,1)

    def forward(self, x):
        return self.linear(x)


xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)

# Data definition
x_data = xy[:, 0:-1]        # 데이터 슬라이싱!  start:end:step 이 하나야! 행, 열, 채널은 쉼표로 구분된다!
y_data = xy[:, -1:]

# Model definition
model = MultivariateLinearRegressionModel()


x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

W = torch.zeros((3,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)        # model.parameters() 함수로 W, b를 리스트로해서 바로 적용시키는듯!

nb_epochs = 20
for epoch in range(nb_epochs+1):

    prediction = model(x_train)             # 디버깅 해보니까 여기를 실행하면 바로 MultivariateLinearRegressionModel의 forward함수가 실행된다! 오오..
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    print('Epoch {:4d}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, cost.item()
    ))