import numpy as np
import copy
from sklearn import datasets

batchUnit = 10 # batch size
max_iter = 100
ERROR_THRESHOLD = 0.00001

def count_sigma(MLP_dictionary, i_layer,node, delta_matrix) :
    sum = 0
    for keys in MLP_dictionary.keys() :
        if keys[0] == i_layer+1 and keys[1]==node :
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
            result_matrix[i][j] = feed_forward_output_matrix[i][j]*(1-feed_forward_output_matrix[i][j]) * count_sigma(MLP_dictionary,i,j,result_matrix)
    return result_matrix
                       
def backward_phase (feed_forward_output_matrix, MLP_dictionary, n_layer, x_data, target_data , delta_weight_dictionary, bias_dictionary) :
    delta_matrix = make_delta_matrix(feed_forward_output_matrix, target_data, MLP_dictionary)
    for k in range (n_layer-1, 0, -1) :
        for keys in MLP_dictionary.keys():
            if k == keys[0]:
                delta_weight_dictionary[keys] += 0.1*delta_matrix[k][keys[2]] * feed_forward_output_matrix[k-1][keys[1]]
    for i in range(0,len(x_data)) :
        for keys in MLP_dictionary.keys():
            if keys[0] == 0 and keys[1]==i:
                delta_weight_dictionary[keys] += 0.1*delta_matrix[0][keys[2]] *x_data[i]
    for key in bias_dictionary.keys() : 
        if key[0]< n_layer-1:
            bias_dictionary[key] = bias_dictionary[key] + 0.1* delta_matrix[key[0]][key[1]]*1 

# network weights in dict data structure
def initWeight(layerNo, inputlayer = 4, outputlayer = 1):
    tempLayerNo = copy.deepcopy(layerNo)
    layerNo.insert(0, inputlayer)
    layerNo.append(outputlayer)
    weight = dict()
    for i in range(len(layerNo)-1):
        for j in range(layerNo[i]):
            for k in range(layerNo[i+1]):
                weight[(i,j,k)] = 0
    return weight, tempLayerNo

def initBias(layerNo, inputlayer = 4, outputlayer = 1):
    tempLayerNo = copy.deepcopy(layerNo)
    tempLayerNo.insert(0, inputlayer)
    tempLayerNo.append(outputlayer)
    bias = dict()
    for i in range(len(tempLayerNo)-1):
        for j in range(tempLayerNo[i+1]):
            bias[(i,j)] = 0
    return bias

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def count_error(target, output):
    error = ((target-output)**2)/2
    return error  

def batching(iris, counter):
    startIdx = counter*batchUnit
    data = iris.data[startIdx:(startIdx)+batchUnit]
    target = iris.target[startIdx:(startIdx)+batchUnit]
    
    return data, target

def forward(record, weight, bias, layerNo, inputlayer = 4):
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
                output[i][k] = sigmoid(output[i][k] + bias[(i, k)])
        else:
            for k in range(int(layerNo[i + 1])):
                for j in range(int(layerNo[i])):
                    output[i][k] += weight[(i, j, k)] * output[i - 1][k - 1]
                output[i][k] = sigmoid(output[i][k] + bias[(i, k)])
    return(output)

def train(iris, weight, bias, layerNo):
    # use batching() in a loop, feeding the batched datasets to the model
    epoch = 1
    error = 999
    while (epoch <= max_iter and error >=ERROR_THRESHOLD  ) :
        error = 0
        for i in range(len(iris.data)//batchUnit):
            delta_weight = copy.deepcopy(weight)
            for keys in delta_weight.keys():
                delta_weight[keys] = 0
            dataTemp, targetTemp = batching(iris, i)
            # manipulate dataTemp and targetTemp here
            for i in range(len(dataTemp)):
                output = forward(dataTemp[i], weight, bias, layerNo)
                backward_phase(output, weight, len(layerNo),dataTemp[i],targetTemp[i], delta_weight, bias)
                error = error + count_error(targetTemp[i],output[len(output)-1][0])
            weight = update_weight(layerNo, weight, delta_weight)
        epoch = epoch + 1
        print(epoch)

    

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
        print("w%d = %g" %(i, values))

def main():
    layerNo = list(map(int, input("Jumlah node tiap layer, dipisahkan dengan space: ").split()))
    weight, layerNoForBias = initWeight(layerNo)
    bias = initBias(layerNoForBias)
    iris = datasets.load_iris() 
    train(iris, weight, bias, layerNo)
    print_model(weight)

if __name__ == "__main__":
    main()