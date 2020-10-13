import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics, datasets, tree
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


def support_vector_machine(x_train, x_test, y_train, kernel_type):
    svmt = svm.SVC(gamma='auto', kernel=kernel_type)
    svmt.fit(x_train, y_train)
    print('svm_'+kernel_type+' training is done')
    return svmt.predict(x_test)


def KNN_claissifier(x_train, x_test, y_train):
    classifier = KNN(n_neighbors=3)
    classifier.fit(x_train, y_train)
    print('KNN training is done')
    return classifier.predict(x_test)


def decision_tree(x_train, x_test, y_train):
    classifier = tree.DecisionTreeClassifier().fit(x_train, y_train)
    print('tree training is done')
    return classifier.predict(x_test)


def forest_classifier(x_train, x_test, y_train):
    classifier = RandomForestClassifier().fit(x_train, y_train)
    print('randomforest training is done')
    return classifier.predict(x_test)


def accuracy_test(class_prediction, y_test, name, data_type):
    ac_score = metrics.accuracy_score(y_test, class_prediction)
    print('['+data_type+'_'+name+']' + f'Accuracy rate = {ac_score:.5f}')

    # wrong = [(p,e) for (p,e) in zip(class_prediction, y_test) if p != e]
    # print(wrong)

    confusion = metrics.confusion_matrix(y_test, class_prediction)
    # print(confusion)
    plt.figure(figsize=(6,5))
    plt.title(name + '\n(accuracy : ' + str(ac_score) + ')')

    if data_type == 'iris':
        confusion_df = pd.DataFrame(confusion, index=['setosa', 'versicolor', 'virginica'], columns=range(3))
        axes = sns.heatmap(confusion_df, annot=True, cmap='nipy_spectral_r')
        plt.savefig('iris_figure/'+name+'.png')
    else:
        confusion_df = pd.DataFrame(confusion, index=list(range(10)), columns=range(10))
        axes = sns.heatmap(confusion_df, annot=True, cmap='nipy_spectral_r')
        plt.savefig('mnist_figure2/'+name+'.png')


def classify(x_train, x_test, y_train, y_test, data_type):
    svm_rbf_pred = support_vector_machine(x_train, x_test, y_train, 'rbf')
    svm_linear_pred = support_vector_machine(x_train, x_test, y_train, 'linear')
    svm_poly_pred = support_vector_machine(x_train, x_test, y_train, 'poly')
    knn_pred = KNN_claissifier(x_train, x_test, y_train)
    forest_pred = forest_classifier(x_train, x_test, y_train)
    tree_pred = decision_tree(x_train, x_test, y_train)

    accuracy_test(svm_rbf_pred, y_test, 'svm_rbf_pred', data_type)
    accuracy_test(svm_poly_pred, y_test, 'svm_poly_pred', data_type)
    accuracy_test(svm_linear_pred, y_test, 'svm_linear_pred', data_type)
    accuracy_test(knn_pred, y_test, 'knn_pred', data_type)
    accuracy_test(forest_pred, y_test, 'forest_pred', data_type)
    accuracy_test(tree_pred, y_test, 'tree_pred', data_type)

if __name__ == '__main__':
    # calssifier using iris dataset
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1011)
    print(x_train.shape, y_train.shape)
    classify(x_train, x_test, y_train, y_test, 'iris')

    # calssifier using mnist dataset
    test = pd.read_csv('dataframes/mnist_test.csv', header=None)
    train = pd.read_csv('dataframes/mnist_train.csv', header=None)[:6000]
    x_train, y_train, x_test, y_test = train.iloc[:,1:].to_numpy(), train.iloc[:,0].to_numpy(), test.iloc[:,1:].to_numpy(), test.iloc[:,0].to_numpy()
    print(x_test.shape, y_test.shape, x_train.shape, y_train.shape)
    classify(x_train, x_test, y_train, y_test, 'mnist')
