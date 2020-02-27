import numpy as np
import copy
from sklearn import datasets

batchUnit = 10 # batch size

def count_sigma(MLP_dictionary, i_layer,node, delta_matrix) :
    sum = 0
    for keys in MLP_dictionary.keys() :
        if keys[0] == i_layer+1 and keys[1]==node :
            print(keys)
            sum = sum + MLP_dictionary[keys] * delta_matrix[i_layer+1][keys[2]]
    return sum

def make_delta_matrix(feed_forward_output_matrix, target_data, MLP_dictionary):
    result_matrix = copy.deepcopy(feed_forward_output_matrix)
    for row in result_matrix :
        for i in range(0,len(row)) :
            row[i] = 0
    result_matrix[len(result_matrix)-1][0] = feed_forward_output_matrix[len(result_matrix)-1][0]*(1-feed_forward_output_matrix[len(result_matrix)-1][0]) * (target_data - feed_forward_output_matrix[len(result_matrix)-1][0]) 
    for i in range(len(feed_forward_output_matrix)-2,-1,-1 ) :
        for j in range(0,len(feed_forward_output_matrix[i])):
            print(i,j)
            result_matrix[i][j] = feed_forward_output_matrix[i][j]*(1-feed_forward_output_matrix[i][j]) * count_sigma(MLP_dictionary,i,j,result_matrix)
    return result_matrix

def backward_phase (feed_forward_output_matrix, MLP_dictionary, n_layer, x_data, target_data , delta_weight_dictionary) :
    delta_matrix = make_delta_matrix(feed_forward_output_matrix, target_data, MLP_dictionary)
    for k in range (n_layer-1, 0, -1) :
        for keys in MLP_dictionary.keys():
            if k == keys[0]  :
                delta_weight_dictionary[keys] += delta_weight_dictionary[keys] +  0.1*delta_matrix[k][keys[2]] * feed_forward_output_matrix[k-1][keys[1]]
    for i in range(0,len(x_data)) :
        for keys in MLP_dictionary.keys():
            if keys[0] == 0 and keys[1]==i:
                delta_weight_dictionary[keys] += delta_weight_dictionary[keys] +  0.1*delta_matrix[0][keys[2]] *x_data[i]

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

def initBias(layerNo, inputlayer = 4, outputlayer = 1):
    layerNo.insert(0, inputlayer)
    layerNo.append(outputlayer)
    bias = dict()
    for i in range(len(layerNo)-1):
        for j in range(layerNo[i+1]):
            bias[(i,j)] = 0
    
    return bias

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def error(target, output):
    error = ((target-output)**2)/2
    return error  

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

def update_weight(layerNo, weight, delta_weight, learning_rate = 0.1):
    for i in range(len(layerNo)-1):
        for j in range(layerNo[i]):
            for k in range(layerNo[i+1]):
                weight[(i,j,k)] = weight[(i,j,k)] - learning_rate * delta_weight[(i,j,k)]
    return weight

def print_model(weight):
    i = 0
    for keys, values in weight.items():
        i += 1
        print("w%d = %.5f" %(i, values))

def main():
    layerNo = list(map(int, input("Jumlah node tiap layer, dipisahkan dengan space: ").split()))
    weight = initWeight(layerNo)
    iris = datasets.load_iris()
    train(iris, weight, layerNo)
    print_model(weight)

if __name__ == "__main__":
    main()