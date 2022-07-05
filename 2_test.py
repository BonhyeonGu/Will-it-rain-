import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def loadTest():
    dataTest_xTEMP = []
    dataTest_x = ["TRASH"]
    dataTest_y = []
    flags = []
    with open("./test.csv", "r") as fr:
        lines = fr.read().split('\n')
    for i in range(1, len(lines) - 1):
        print("\rProcess : %s : %.4f%%" % ("./test.csv", (float(i) / (len(lines) - 2)) * 100), end="")
        line = lines[i].split(',')
        data_one = [int(line[0]), int(line[2].split('-')[1])]
        flags.append([line[0], line[2]])
        breakChain = False
        for j in range(3, 8):
            if line[j] == '':
                breakChain = True
                break
            data_one.append(float(line[j]))
        if breakChain:
            continue
        for j in range(9, 36):
            if line[j] == '':
                data_one.append(0.0)
            else:
                data_one.append(float(line[j]))
        if (line[8] == ''):
            data_one.append(0.0)
        else:          
            data_one.append(1.0)
        dataTest_xTEMP.append(data_one)
    frontId = "x"
    nowId = "x"
    c = 1
    for tempOne in dataTest_xTEMP:
        nowId = tempOne[0]
        if frontId != nowId:
            dataTest_x.pop()
        else:
            dataTest_y.append([tempOne[34]])
        dataTest_x.append(tempOne[:34])
        frontId = nowId
        c += 1
    dataTest_x.pop()
    mins = np.load("./mins.npy")
    maxs = np.load("./maxs.npy")
    c = 1
    for dataTest_x_one in dataTest_x:
        for i in range(len(dataTest_x_one)):
            dataTest_x_one[i] = (dataTest_x_one[i] - mins[i]) / (maxs[i] - mins[i])
        c += 1
    return dataTest_x, dataTest_y, flags

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

ok = 0
model = MLP()
model.load_state_dict(torch.load("mlp.pth"))
x_test, y_test, flags = loadTest()
with torch.no_grad():
    model.eval()
    for i in range(0, len(x_test)):
        prediction = model(torch.FloatTensor([x_test[i]]))
        print(flags[i])
        print("예측 %f, 실제 %f"%(prediction, y_test[i][0]))
        if (prediction >= 0.5 and y_test[i][0] == 1.0) or (prediction < 0.5 and y_test[i][0] == 0.0):
            print("성공")
            ok += 1
        else:
            print("실패")
print("\n 정확도는 다음과 같습니다. : %f%%" % (ok / len(x_test) * 100))