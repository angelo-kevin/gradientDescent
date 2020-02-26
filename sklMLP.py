from sklearn.neural_network import MLPClassifier
from sklearn import datasets

def main():
    layerNo = list(map(int, input("Jumlah node tiap layer, dipisahkan dengan space: ").split()))

    iris = datasets.load_iris()
    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=tuple(layerNo), activation='logistic', max_iter=1000)
    clf.fit(iris.data, iris.target)

    for i in clf.coefs_:
        print(i) # returns the weight of the network

if __name__ == "__main__":
    main()