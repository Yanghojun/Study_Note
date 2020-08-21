import torch
import torch.nn as nn
inputs = torch.Tensor(1,1,28,28)
print(inputs)
print(inputs.shape)

conv1=nn.Conv2d(1,32,3,padding=1)
pool = nn.MaxPool2d(2)
conv2=nn.Conv2d(32,64,3,padding=1)

print(conv1)
print(conv2)
print(pool)

out = conv1(inputs)
print(out.shape)

out = pool(out)
print(out.shape)

out = conv2(out)
print(out.shape)

out = pool(out)
print(out.shape)

print(out.size(0))      # 텐서의 속성을 뽑아올 수 있는 것임. (이건 batchsize)
print(out.size(1))      # Channel
print(out.size(2))      # Height
print(out.size(3))      # Width

out = out.view(out.size(0), -1)     # 배치사이즈만 그대로 남겨두고 나머지는 일자로 쭉 펼친다!. 아하 이게 Fully connected랑 연관되어 있구먼
print(out.shape)

fc = nn.Linear(3136,10)
out = fc(out)
print(out.shape)