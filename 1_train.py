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
        self.fc1 = nn.Linear(in_features=34, out_features=80, bias=True)
        self.fc2 = nn.Linear(in_features=80, out_features=30, bias=True)
        self.fc3 = nn.Linear(in_features=30, out_features=1, bias=True)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.relu(self.fc3(out))
        return out

batch = 100
epochs = 20#전체 data에서 몇 번 반복할 것인지
learning_rate = 1e-5
dataset = CustomDataset()
dataloader = DataLoader(dataset=dataset, batch_size=batch, shuffle=True, drop_last=True)#data set, batch size, training_epochs 번 data를 섞겠다, 남은건 버린다.

loss_function = nn.CrossEntropyLoss()#loss function
model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-7)#어떤 wp를 업데이트 할 것인가
criterion = nn.MSELoss()

for epoch in range(epochs + 1):
    runningLoss = 0.0
    for idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        optimizer.zero_grad()
        pred = model(x_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        runningLoss += loss.item()
    print("loss : %f " %(runningLoss/len(dataloader)))
torch.save(model.state_dict(), "mlp.pth")#wp 저장
print('Learning finished')