from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from math import sqrt
import numpy as np
import pandas as pd


# returns a normalised np.array of the dataset values, the dataset column of the dependent
# variable with values {Benign, Malignant} and the dataset labels
def pre_processing():
    # code from teaching material
    breast = load_breast_cancer()
    breast_data = breast.data
    breast_labels = breast.target
    labels = np.reshape(breast_labels, (569, 1))
    final_breast_data = np.concatenate([breast_data, labels], axis=1)
    breast_dataset = pd.DataFrame(final_breast_data)
    features = breast.feature_names
    features_labels = np.append(features, 'label')
    breast_dataset.columns = features_labels
    breast_dataset['label'].replace(0, 'Benign', inplace=True)
    breast_dataset['label'].replace(1, 'Malignant', inplace=True)
    x = breast_dataset.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    feat_cols = ['feature' + str(feat) for feat in range(x.shape[1])]
    labels = breast_dataset['label']
    return x, labels, feat_cols


# selects the appropriate number of principal components
# based on the total explained variance
def select(d):
    c = []
    pc = np.empty(1)
    for i in range(1, d.shape[1] + 1):
        pca = PCA(n_components=i)
        pc = pca.fit_transform(d)
        c.append('principal component {}'.format(i))
        if pca.explained_variance_ratio_.sum() >= 0.75:
            break
    return pc, c


# performs a logistic regression with stratified sampling and returns
# the predicted result along with the corresponding test data
def reg(df, column):
    x_train, x_test, y_train, y_test = train_test_split(df, column, stratify=column, test_size=0.15)
    log = LogisticRegression(solver='liblinear')
    log.fit(x_train, y_train)
    predictions = log.predict(x_test)
    return predictions, np.array(y_test)


# examines the existence of statistically significant differences between two models
def compare(m1, t1, m2, t2):
    # use of sklearn's accuracy_score() to calculate classification error
    # alternatively use the accuracy(model, true) function below for that same purpose
    e1 = 1 - accuracy_score(t1, m1)
    e2 = 1 - accuracy_score(t2, m2)
    v1, v2 = e1*(1-e1), e2*(1-e2)
    p = abs(e1-e2)/sqrt((v1+v2)/(len(m1)))
    print('test instances n = {}\n\noriginal model metrics\ne1 = {}, v1 = {}\n\nprincipal component metrics\ne2 = {}, '
          'v2 = {}\n'.format(len(m1), e1, v1, e2, v2))
    message = 'H1 holds'
    if p < 2:
        message = 'H0 is true'
    print('p = {}\n{}'.format(p, message))


# computes classifier accuracy
def accuracy(model, true):
    count = 0
    for i in range(len(model)):
        if model[i] == true[i]:
            count += 1
    total = len(model)
    count = count / total
    return count


data, label, n_columns = pre_processing()
normalised_df = pd.DataFrame(data=data, columns=n_columns)
# print(normalised_df.tail())  # debugging print
principal_components, p_columns = select(data)
principal_df = pd.DataFrame(data=principal_components, columns=p_columns)
# print(principal_df.tail())  # debugging print
p_predictions, p_validation = reg(principal_df, label)
n_predictions, n_validation = reg(normalised_df, label)
compare(n_predictions, n_validation, p_predictions, p_validation)
