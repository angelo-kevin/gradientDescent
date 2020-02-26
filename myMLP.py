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
                weight[(i,j,k)] = 0
    
    return weight

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def batching(iris, counter):
    startIdx = counter*batchUnit
    data = iris.data[startIdx:(startIdx)+batchUnit]
    target = iris.target[startIdx:(startIdx)+batchUnit]
    
    return data, target

def train(iris):
    # use batching() in a loop, feeding the batched datasets to the model
    for i in range(len(iris.data)//batchUnit):
        dataTemp, targetTemp = batching(iris, i)
        # manipulate dataTemp and targetTemp here

def main():
    layerNo = list(map(int, input("Jumlah node tiap layer, dipisahkan dengan space: ").split()))
    weight = initWeight(layerNo)
    iris = datasets.load_iris()
    train(iris)

if __name__ == "__main__":
    main()