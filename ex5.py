from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def clean_penguins():
    file_names = ['PenguinData/adelie.csv', 'PenguinData/chinstrap.csv', 'PenguinData/gentoo.csv']
    data_all = pd.concat((pd.read_csv(i) for i in file_names)).reset_index(drop=True)
    irrelevant_columns = ['studyName', 'Sample Number', 'Region', 'Individual ID', 'Stage', 'Date Egg', 'Comments']

    dict_Clutch_Completion = {'Yes': 1, 'No': 0}
    data_all['Clutch Completion'] = data_all['Clutch Completion'].map(dict_Clutch_Completion)

    dict_Sex = {'MALE': 1, 'FEMALE': 0}
    data_all['Sex'] = data_all['Sex'].map(dict_Sex)

    dict_Species = {'Adelie Penguin (Pygoscelis adeliae)': 0,
                    'Chinstrap penguin (Pygoscelis antarctica)': 1,
                    'Gentoo penguin (Pygoscelis papua)': 2}
    data_all['Species'] = data_all['Species'].map(dict_Species)

    dict_Islands = {'Torgersen': 0,
                    'Biscoe': 1,
                    'Dream': 2}
    data_all['Island'] = data_all['Island'].map(dict_Islands)
    data_all = data_all.drop(columns=irrelevant_columns)

    data_all = data_all.dropna()
    data_all.to_csv('penguins.csv', index=False)


def run_classifier(classifier):
    CV = KFold(n_splits=K)
    f1_scores = np.zeros(K)
    index = 0
    for train, test in CV.split(X):
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]

        model = classifier.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        f1_scores[index] = metrics.f1_score(y_test, y_pred, average='weighted')
        index += 1

    return np.mean(f1_scores)


# clean_penguins()

data = pd.read_csv("penguins.csv")

X = data.loc[:, data.columns != 'Species'].to_numpy()
y = data.loc[:, data.columns == 'Species'].to_numpy().ravel()

K = 10

plt.figure(figsize=(7, 7))

# Analysis with n_estimators
estimators = np.arange(30, 150, 10)
f1_score_per_param = np.zeros(len(estimators))
for i, nr_estimators in enumerate(estimators):
    f1_score_per_param[i] = run_classifier(RandomForestClassifier(n_estimators=nr_estimators))

plt.plot(estimators, f1_score_per_param, c='g')
plt.xlabel('number of estimators')
plt.ylabel('Mean f1 score')
plt.title('f1 with different number of estimators')
plt.show()

# Comparison between ensemble and non-ensemble
svc = SVC(probability=True, kernel='linear')
rfc = RandomForestClassifier(n_estimators=50)
svc_f1 = run_classifier(svc)
rfc_f1 = run_classifier(rfc)
print('f1 score for non ensemble: {:.3f} vs f1 score for ensemble: {:.3f}'.format(svc_f1, rfc_f1))
