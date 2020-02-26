from sklearn.neural_network import MLPClassifier
from sklearn import datasets

def main():
    iris = datasets.load_iris()
    clp = MLPClassifier()

if __name__ == "__main__":
    main()