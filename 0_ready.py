import pickle
import os
import numpy as np
dir = './Ready/'
dataTrain_xTEMP = []
dataTrain_x = ["TRASH"]
dataTrain_y = []
idToName = dict()
nameToId = dict()
compliteId = set()

print("Parsing, Merge")
fileNames = os.listdir(dir)
for fileName in fileNames:
    with open(dir + fileName, "r") as fr:
        lines = fr.read().split('\n')
    for i in range(1, len(lines) - 1):
        print("\rProcess : %s : %.4f%%" % (fileName, (float(i) / (len(lines) - 2)) * 100), end="")
        line = lines[i].split(',')
        if not (line[0] in compliteId):
            compliteId.add(line[0])
            idToName[line[0]] = line[1]
            nameToId[line[1]] = line[0]
        data_one = [int(line[0]), int(line[2].split('-')[1])]
        breakChain = False
        for j in range(3, 8):
            #잘못된 케이스 = 버림
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
        data_one.append(line[2])
        dataTrain_xTEMP.append(data_one)

print("\nSort")
#임시 x는 [0:id, ~ 34:y, 35:2012-05-31] 형식이다.
dataTrain_xTEMP = sorted(dataTrain_xTEMP, key=lambda x: (x[0], x[35]))
print("\nSplit")
frontId = "x"
nowId = "x"
c = 1
for tempOne in dataTrain_xTEMP:
    print("\rProcess : %.4f%%" % ((float(c) / (len(dataTrain_xTEMP))) * 100), end="")
    nowId = tempOne[0]
    if frontId != nowId:
        dataTrain_x.pop()
    else:
        dataTrain_y.append([tempOne[34]])
    dataTrain_x.append(tempOne[:34])
    frontId = nowId
    c += 1
dataTrain_x.pop()

print("\nNormalization")
maxs = np.empty(34)
mins = np.empty(34)
maxs.fill(-1.0)
mins.fill(np.Inf)
c = 1
for dataTrain_x_one in dataTrain_x:
    print("\rProcess : Find max min : %.4f%%" % ((float(c) / (len(dataTrain_x))) * 100), end="")
    for i in range(len(dataTrain_x_one)):
        if maxs[i] < dataTrain_x_one[i]:
            maxs[i] = dataTrain_x_one[i]
        if mins[i] > dataTrain_x_one[i]:
            mins[i] = dataTrain_x_one[i]
    c += 1
c = 1
for dataTrain_x_one in dataTrain_x:
    print("\rProcess : Normalization : %.4f%%" % ((float(c) / (len(dataTrain_x))) * 100), end="")
    for i in range(len(dataTrain_x_one)):
        dataTrain_x_one[i] = (dataTrain_x_one[i] - mins[i]) / (maxs[i] - mins[i])
    c += 1
            
print("\nSave")
with open("./dataTrain_x", "wb") as fw:
    pickle.dump(dataTrain_x, fw)
with open("./dataTrain_y", "wb") as fw:
    pickle.dump(dataTrain_y, fw)
with open("./idToName", "wb") as fw:
    pickle.dump(idToName, fw)
with open("./nameToId", "wb") as fw:
    pickle.dump(nameToId, fw)
np.save("./maxs.npy", maxs)
np.save("./mins.npy", mins)
input("\nEnd")