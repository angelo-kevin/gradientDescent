import numpy as np
import copy
from sklearn import datasets
import networkx as nx
import matplotlib.pyplot as plt

batchUnit = 1 # batch size
max_iter = 1000
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
                       
def backward_phase (feed_forward_output_matrix, MLP_dictionary, n_layer, x_data, target_data , delta_weight_dictionary, delta_bias_dictionary) :
    delta_matrix = make_delta_matrix(feed_forward_output_matrix, target_data, MLP_dictionary)
    for k in range (n_layer-1, 0, -1) :
        for keys in MLP_dictionary.keys():
            if k == keys[0]:
                delta_weight_dictionary[keys] += 0.1*delta_matrix[k][keys[2]] * feed_forward_output_matrix[k-1][keys[1]]
    for i in range(0,len(x_data)) :
        for keys in MLP_dictionary.keys():
            if keys[0] == 0 and keys[1]==i:
                delta_weight_dictionary[keys] += 0.1*delta_matrix[0][keys[2]] *x_data[i]
    for key in delta_bias_dictionary.keys() : 
        if key[0]< n_layer-1:
            delta_bias_dictionary[key] = delta_bias_dictionary[key] + 0.1* delta_matrix[key[0]][key[1]]*1 

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
                    # print(i, j, k)
                    # print(output[i][k])
                    # print(weight[(i, j, k)])
                    # print(record[j])
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
    delta_weight = copy.deepcopy(weight)
    delta_bias = copy.deepcopy(bias)
    while (epoch <= max_iter and error >=ERROR_THRESHOLD):
        out = []
        error = 0
        for i in range(len(iris.data)//batchUnit):
            for key in delta_bias.keys() :
                delta_bias[key] = 0
            for key in delta_weight.keys():
                delta_weight[key] = 0
            dataTemp, targetTemp = batching(iris, i)
            # manipulate dataTemp and targetTemp here
            for i in range(len(dataTemp)):
                output = forward(dataTemp[i], weight, bias, layerNo)
                out.append(output[len(output)-1][0])
                backward_phase(output, weight, len(layerNo),dataTemp[i],targetTemp[i], delta_weight, delta_bias)
                error = error + count_error(targetTemp[i],output[len(output)-1][0])
            weight = update_weight(layerNo, weight, delta_weight)
            bias = update_bias(bias, delta_bias)
        epoch = epoch + 1
    print(out)

    

def update_weight(layerNo, weight, delta_weight, learning_rate = 0.1):
    for i in range(len(layerNo)-1):
        for j in range(layerNo[i]):
            for k in range(layerNo[i+1]):
                weight[(i,j,k)] = weight[(i,j,k)] - learning_rate * delta_weight[(i,j,k)]
    return weight

def update_bias(bias, delta_bias, learning_rate = 0.1):
    for keys in delta_bias.keys():
        bias[keys] += delta_bias[keys]
    return bias

def print_model(layerNo, weight, inputlayer = 4, outputlayer = 1):
    G = nx.Graph()
    pos_layout = {}
    for i in range(len(layerNo)-1):
        for k in range(layerNo[i+1]):
            for j in range(layerNo[i]):
                if (i==0):
                    source_node = "i"+str(j+1)
                    target_node = "h"+str(i+1)+"."+str(k+1)
                elif (i==len(layerNo) - 2):
                    source_node = "h"+str(i)+"."+str(j+1)
                    target_node = "o"+str(k + 1)
                else:
                    source_node = "h"+str(i)+"."+str(j+1)
                    target_node = "h"+str(i+1)+"."+str(k+1)
                G.add_edge(source_node, target_node, weight=round(weight[(i,j,k)], 5))
                pos_layout[source_node] = [i, j]
                pos_layout[target_node] = [i+1, k]
    layout = pos_layout
    nx.draw(G, layout)
    nx.draw_networkx_nodes(G, layout)
    nx.draw_networkx_labels(G, pos=layout)
    nx.draw_networkx_edge_labels(G, pos=layout)
    plt.show()

def predict(data, weight1, bias1, weight2, bias2, layerNo):
    output = []
    for record in data.data:
        print(record)
        out = forward(record, weight1, bias1, layerNo)
        if (out[len(out)-1][0] < 0.5):
            output.append(0)
        else:
            out = forward(record, weight2, bias2, layerNo)
            if (out[len(out)-1][0] < 0.5):
                output.append(1)
            else:
                output.append(2)
    return output

def main():
    layerNo = list(map(int, input("Jumlah node tiap layer, dipisahkan dengan space: ").split()))
    weight1, layerNoForBias = initWeight(layerNo)
    bias1 = initBias(layerNoForBias)
    weight2 = copy.deepcopy(weight1)
    bias2 = copy.deepcopy(bias1)
    iris1 = datasets.load_iris()
    iris2 = datasets.load_iris()

    # Target: Setosa
    iris1.target = (iris1.target==0).astype(np.int8)
    print(iris1.target)
    train(iris1, weight1, bias1, layerNo)

    # Target: Versicolor
    iris2.target = (iris2.target==1).astype(np.int8)
    print(iris2.target)
    train(iris2, weight2, bias2, layerNo)

    # Test
    output_test = predict(iris1, weight1, bias1, weight2, bias2, layerNo)
    print(output_test)

    print_model(layerNo, weight1)
    print_model(layerNo, weight2)

if __name__ == "__main__":
    main()