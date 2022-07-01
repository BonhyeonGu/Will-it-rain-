import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def loadTest():
    dataTest_xTEMP = []
    dataTest_x = ["TRASH"]
    dataTest_y = []
    with open("./test.csv", "r") as fr:
        lines = fr.read().split('\n')
    for i in range(1, len(lines) - 1):
        print("\rProcess : %s : %.4f%%" % ("./test.csv", (float(i) / (len(lines) - 2)) * 100), end="")
        line = lines[i].split(',')
        data_one = [int(line[0]), int(line[2].split('-')[1])]
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
    return dataTest_x, dataTest_y

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

network = MLP()
network.load_state_dict(torch.load("mlp.pth"))

x_test, y_test = loadTest()
with torch.no_grad():
    for i in range(0, len(x_test)):
        prediction = network(torch.FloatTensor([x_test[i]]))
        print("예측 값은 %f 입니다."%(prediction))
        print("실제 값은 %f 입니다."%(y_test[i][0]))