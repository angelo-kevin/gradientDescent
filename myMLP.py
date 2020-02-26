import numpy as np
from sklearn import datasets

batchUnit = 10 # batch size

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
    iris = datasets.load_iris()
    train(iris)

if __name__ == "__main__":
    main()