import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pickle

class CustomDataset(Dataset): 
    def __init__(self):
        with open("dataTrain_x", 'rb') as f:
            self.x_data = pickle.load(f)
        with open("dataTrain_y", 'rb') as f:
            self.y_data = pickle.load(f)

    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=34, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=8)
        self.fc3 = nn.Linear(in_features=8, out_features=1)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        out = self.sig(self.fc1(x))
        out = self.sig(self.fc2(out))
        out = self.fc3(out)
        return out

batch_size = 10
learning_rate = 0.1
training_epochs = 15#전체 data에서 몇 번 반복할 것인지
loss_function = nn.CrossEntropyLoss()#loss function
model = MLP()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)#어떤 wp를 업데이트 할 것인가
data_loader = DataLoader(dataset=CustomDataset(), batch_size=batch_size, shuffle=True, drop_last=False)#data set, batch size, training_epochs 번 data를 섞겠다, 남은건 버린다.

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)
    for x, y in data_loader:
        pred = model(x)#실행, 결과받기
        cost = F.mse_loss(pred, y)
        # cost로 H(x) 계산
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        avg_cost += cost.item() / total_batch #epoch 끝날때마다 출력
    print('Epoch: %d Loss = %f'%(epoch+1, avg_cost))
torch.save(model.state_dict(), "mlp.pth")#wp 저장
print('Learning finished')