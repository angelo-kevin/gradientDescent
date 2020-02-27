import numpy as np
from sklearn import datasets

batchUnit = 10 # batch size

# network weights in dict data structure
def initWeight(layerNo, inputlayer = 4, outputlayer = 1):
    layerNo.insert(0, inputlayer)
    layerNo.append(outputlayer)
    weight = dict()
    for i in range(len(layerNo)-1):
        for j in range(layerNo[i]):
            for k in range(layerNo[i+1]):
                weight[(i,j,k)] = 1
    
    return weight

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def batching(iris, counter):
    startIdx = counter*batchUnit
    data = iris.data[startIdx:(startIdx)+batchUnit]
    target = iris.target[startIdx:(startIdx)+batchUnit]
    print(data)
    
    return data, target

def forward(record, weight, layerNo, inputlayer = 4):
    # Initialize output matrix
    output = []
    for i in range(len(layerNo) - 2):
        output.append([])
        for k in range(layerNo[i + 1]):
            output[i].append(0)
    output.append([0])  # Output layer

    # Calculate out
    for i in range(len(layerNo) - 1):
        if (i == 0):
            for k in range(int(layerNo[i + 1])):
                for j in range(len(record)):
                    output[i][k] += weight[(i, j, k)] * record[j]
                output[i][k] = sigmoid(output[i][k])
        else:
            for k in range(int(layerNo[i + 1])):
                for j in range(int(layerNo[i])):
                    output[i][k] += weight[(i, j, k)] * output[i - 1][k - 1]
                output[i][k] = sigmoid(output[i][k])
    
    return(output)

def train(iris, weight, layerNo):
    # use batching() in a loop, feeding the batched datasets to the model
    for i in range(len(iris.data)//batchUnit):
        dataTemp, targetTemp = batching(iris, i)
        # manipulate dataTemp and targetTemp here
        for record in dataTemp:
            forward(record, weight, layerNo)

def main():
    layerNo = list(map(int, input("Jumlah node tiap layer, dipisahkan dengan space: ").split()))
    weight = initWeight(layerNo)
    iris = datasets.load_iris()
    train(iris, weight, layerNo)

if __name__ == "__main__":
    main()