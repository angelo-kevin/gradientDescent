import numpy as np
from sklearn import datasets

batchUnit = 10 # batch size

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def batching(iris, counter):
    data = iris.data[counter*batchUnit:(counter*batchUnit)+batchUnit]
    target = iris.target[counter*batchUnit:(counter*batchUnit)+batchUnit]
    
    return data, target

def main():
    iris = datasets.load_iris()
    # use batching() in a loop, feeding the batched datasets to the model
    for i in range(len(iris.data)//batchUnit):
        dataTemp, targetTemp = batching(iris, i)
        # manipulate dataTemp and targetTemp here

if __name__ == "__main__":
    main()