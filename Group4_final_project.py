import pandas as pd
import sklearn

from statistics import mode
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

credit = pd.read_csv('credit_default.csv')
y = credit['DEFAULT_NEXT_MONTH'].to_numpy()
X = credit.drop(columns = ['DEFAULT_NEXT_MONTH']).to_numpy()
null_records = credit[credit.isnull().any(axis=1)]
print(f"Number of records with nulls: {len(null_records)}")


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=20)
normalizer = sklearn.preprocessing.MinMaxScaler()
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm = normalizer.transform(X_test)


#Random Forest Classifier
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth = 10, min_samples_leaf=4, min_samples_split=5, class_weight= 'balanced', random_state=42)
clf.fit(X_train, y_train)
y_pred_test = clf.predict(X_test)
print('predicted:', y_pred_test)


accuracy = sklearn.metrics.accuracy_score(y_true=y_test, y_pred = y_pred_test)
precision = sklearn.metrics.precision_score(y_true=y_test, y_pred = y_pred_test)
recall = sklearn.metrics.recall_score(y_true=y_test, y_pred = y_pred_test)
f1 = sklearn.metrics.f1_score(y_true=y_test, y_pred = y_pred_test)
print('Random Forest')
print('Accuracy:', accuracy, 'Precision:', precision, 'Recall:',recall, 'F1-Score:',f1)

#Majority baseline
y_pred_maj = [mode(y_train)]*len(y_test) # y_pred_maj=0 due to the majority of the dataset
accuracy_maj = sklearn.metrics.accuracy_score(y_true=y_test, y_pred = y_pred_maj)
precision_maj = sklearn.metrics.precision_score(y_true=y_test, y_pred = y_pred_maj)
recall_maj = sklearn.metrics.recall_score(y_true=y_test, y_pred = y_pred_maj)
f1_maj = sklearn.metrics.f1_score(y_true=y_test, y_pred = y_pred_maj)
print('Majority Baseline')
print('Accuracy:', accuracy_maj, 'Precision:', precision_maj, 'Recall:',recall_maj, 'F1-Score:',f1_maj)