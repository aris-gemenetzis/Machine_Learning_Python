import pandas as pd
import pydotplus
from openpyxl import load_workbook
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.externals.six import StringIO
from IPython.display import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def pre_processing():
    wb = load_workbook('Cardiology.xlsx')
    ws = wb['Sheet1']
    data = ws.values
    columns = next(data)[0:]
    df = pd.DataFrame(data, columns=columns)
    # print(df, df.shape)  # debugging print
    ohe = ['sex', 'chest pain type', 'resting ecg', 'thal']
    ohe_part = one_hot_encoding(df, ohe)
    le = ['slope', 'Fasting blood sugar <120', 'angina']
    ldf = pd.DataFrame(df[le])
    le_part = label_encoding(ldf, le)
    # print('head\n', ldf.head())  # debugging print
    n_list = ['age', 'blood pressure', 'cholesterol', 'maximum heart rate', 'peak', '#colored vessels']
    n_part = pd.DataFrame(df[n_list])
    n_part.rename(columns={'#colored vessels': 'Number of colored vessels'}, inplace=True)
    n_part.rename(columns={i: i.capitalize() for i in n_list if i[0].islower()}, inplace=True)
    # decision trees do not require normalisation 
    # so numercal variables are left as is (not normalised)
    df['class'] = [i.capitalize() for i in df['class']]  # capitalising dependent variable values
    frames = [ohe_part, le_part, n_part, df['class']]
    final = pd.concat(frames, axis=1, sort=False)
    final.rename(columns={'class': 'Class'}, inplace=True)
    return final


# decided to one-hot encode the sex, chest pain type, resting ecg & thal variables
# bc there didn't seem to be any meaningful relative order between the values of each category
def one_hot_encoding(data, ohe_list):
    ohe_dict = {i: (list(data[i].unique()), data.columns.get_loc(i)) for i in ohe_list}
    # print(ohe_dict)  # debugging print
    ohe_pre_process = ColumnTransformer([('dummy', OneHotEncoder(categories=[ohe_dict.get(i)[0] for i in ohe_dict]),
                                          [ohe_dict.get(i)[1] for i in ohe_dict])])
    o = ohe_pre_process.fit_transform(data)
    ohe_data = pd.DataFrame(o.astype(int))  # thought int values would be more cohesive
    c = [[i.capitalize()+': '+s for s in j[0]] for i, j in ohe_dict.items()]
    c = [item for sublist in c for item in sublist]  # just flattening c w/out numpy or itertools
    ohe_data.columns = c
    # print(ohe_data)  # debugging print
    ohe_data.rename(columns={'Chest pain type:  Asymptomatic': 'Chest pain type: Asymptomatic'}, inplace=True)
    return ohe_data


# decided to label encode the slope variable
# because its values seem to have ordinal properties (Down, Flat, Up) wrt heart rate slope
# also decided to label encode the fasting blood sugar & angina variables bc they're binary
def label_encoding(data, le_list):
    for i in data[le_list]:
        data[i] = data[i].astype('category').cat.codes  # assigns values alphabetically
    data.rename(columns={i: i.capitalize() for i in le_list if i[0].islower()}, inplace=True)
    # print(data.head())
    return data


# assuming we're more interested on whether someone is sick rather than healthy the tree classifier can
# do the encoding for the dependent variable on its own, it should map 'healthy' to position 0 & 'sick' to position 1
# based on alphabetical order
def classify(data):
    # split dataset in features and target variable
    x = data[data.columns[:-1]]
    # print('x\n', x, '\n', x.shape)  # debugging print
    y = data[data.columns[-1]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.25)
    # entropy criterion used due to being the preferred measure for node impurity evaluation
    cl = DecisionTreeClassifier(criterion='entropy', max_depth=3)
    # set max tree depth at 3
    cl = cl.fit(x_train, y_train)
    y_pred = cl.predict(x_test)
    return y_pred, y_test, cl


def plot_tree(t, df, name):
    data = StringIO()
    export_graphviz(t, out_file=data, filled=True, rounded=True, special_characters=True,
                    feature_names=[i for i in list(df.columns[:-1])], class_names=['0 (Healthy)', '1 (Sick)'])
    graph = pydotplus.graph_from_dot_data(data.getvalue())
    graph.write_png(name)
    Image(graph.create_png())


def plot_matrix(y_test, y_pred, method):
    # additional accuracy metric used as a helpful indicator for evaluation
    print(method, 'accuracy:', accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm,
                         index=['Healthy', 'Sick'],
                         columns=['Predicted Healthy', 'Predicted Sick'])
    print(cm_df)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True)
    plt.title('Cardiology data \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
    plt.ylabel('True values')
    plt.xlabel('Predicted values')
    plt.show()


cardio = pre_processing()
print(cardio.head())
# print(cardio.columns, len(list(cardio.columns)))  # debugging print
predictions, test, tree = classify(cardio)
plot_tree(tree, cardio, 'cardio.png')
plot_matrix(test, predictions, 'mixed encoding')

