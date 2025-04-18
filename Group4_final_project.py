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
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100,class_weight= 'balanced')
clf.fit(X_train, y_train)
y_pred_test = clf.predict(X_test)
accuracy = sklearn.metrics.accuracy_score(y_true=y_test, y_pred = y_pred_test)
precision = sklearn.metrics.precision_score(y_true=y_test, y_pred = y_pred_test)
recall = sklearn.metrics.recall_score(y_true=y_test, y_pred = y_pred_test)
f1 = sklearn.metrics.f1_score(y_true=y_test, y_pred = y_pred_test)
print('\nRandom Forest 1 (n_estimators=100,class_weight=balanced)')
print('Accuracy:', accuracy, 'Precision:', precision, 'Recall:',recall, 'F1-Score:',f1)


clf2 = sklearn.ensemble.RandomForestClassifier(n_estimators=100,class_weight= None)
clf2.fit(X_train, y_train)
y_pred_test2 = clf2.predict(X_test)
accuracy2 = sklearn.metrics.accuracy_score(y_true=y_test, y_pred = y_pred_test2)
precision2 = sklearn.metrics.precision_score(y_true=y_test, y_pred = y_pred_test2)
recall2 = sklearn.metrics.recall_score(y_true=y_test, y_pred = y_pred_test2)
f1_2 = sklearn.metrics.f1_score(y_true=y_test, y_pred = y_pred_test2)
print('\nRandom Forest 2 (n_estimators=100,class_weight= None)')
print('Accuracy:', accuracy2, 'Precision:', precision2, 'Recall:',recall2, 'F1-Score:',f1_2)


clf3 = sklearn.ensemble.RandomForestClassifier(n_estimators=200,class_weight= None)
clf3.fit(X_train, y_train)
y_pred_test3 = clf3.predict(X_test)
accuracy3 = sklearn.metrics.accuracy_score(y_true=y_test, y_pred = y_pred_test3)
precision3 = sklearn.metrics.precision_score(y_true=y_test, y_pred = y_pred_test3)
recall3 = sklearn.metrics.recall_score(y_true=y_test, y_pred = y_pred_test3)
f1_3 = sklearn.metrics.f1_score(y_true=y_test, y_pred = y_pred_test3)
print('\nRandom Forest 3 (n_estimators=200,class_weight= None)')
print('Accuracy:', accuracy3, 'Precision:', precision3, 'Recall:',recall3, 'F1-Score:',f1_3)


clf4 = sklearn.ensemble.RandomForestClassifier(n_estimators=200,class_weight= 'balanced')
clf4.fit(X_train, y_train)
y_pred_test4 = clf4.predict(X_test)
accuracy4 = sklearn.metrics.accuracy_score(y_true=y_test, y_pred = y_pred_test4)
precision4 = sklearn.metrics.precision_score(y_true=y_test, y_pred = y_pred_test4)
recall4 = sklearn.metrics.recall_score(y_true=y_test, y_pred = y_pred_test4)
f1_4 = sklearn.metrics.f1_score(y_true=y_test, y_pred = y_pred_test4)
print('\nRandom Forest 4 (n_estimators=200,class_weight=balanced)')
print('Accuracy:', accuracy4, 'Precision:', precision4, 'Recall:',recall4, 'F1-Score:',f1_4)


clf5 = sklearn.ensemble.RandomForestClassifier(n_estimators=200, criterion = 'gini', class_weight= 'balanced')
clf5.fit(X_train, y_train)
y_pred_test5 = clf5.predict(X_test)
accuracy5 = sklearn.metrics.accuracy_score(y_true=y_test, y_pred = y_pred_test5)
precision5 = sklearn.metrics.precision_score(y_true=y_test, y_pred = y_pred_test5)
recall5 = sklearn.metrics.recall_score(y_true=y_test, y_pred = y_pred_test5)
f1_5 = sklearn.metrics.f1_score(y_true=y_test, y_pred = y_pred_test5)
print('\nRandom Forest 5 (n_estimators=200, criterion = gini, class_weight= balanced)')
print('Accuracy:', accuracy5, 'Precision:', precision5, 'Recall:',recall5, 'F1-Score:',f1_5)


clf6 = sklearn.ensemble.RandomForestClassifier(n_estimators=200, criterion = 'entropy', class_weight= 'balanced')
clf6.fit(X_train, y_train)
y_pred_test6 = clf6.predict(X_test)
accuracy6 = sklearn.metrics.accuracy_score(y_true=y_test, y_pred = y_pred_test6)
precision6 = sklearn.metrics.precision_score(y_true=y_test, y_pred = y_pred_test6)
recall6 = sklearn.metrics.recall_score(y_true=y_test, y_pred = y_pred_test6)
f1_6 = sklearn.metrics.f1_score(y_true=y_test, y_pred = y_pred_test6)
print('\nRandom Forest 6 (n_estimators=200, criterion = entropy, class_weight= balanced)')
print('Accuracy:', accuracy6, 'Precision:', precision6, 'Recall:',recall6, 'F1-Score:',f1_6)


#Majority baseline
y_pred_maj = [mode(y_train)]*len(y_test) # y_pred_maj=0 due to the majority of the dataset
accuracy_maj = sklearn.metrics.accuracy_score(y_true=y_test, y_pred = y_pred_maj)
precision_maj = sklearn.metrics.precision_score(y_true=y_test, y_pred = y_pred_maj)
recall_maj = sklearn.metrics.recall_score(y_true=y_test, y_pred = y_pred_maj)
f1_maj = sklearn.metrics.f1_score(y_true=y_test, y_pred = y_pred_maj)
print('\nMajority Baseline')
print('Accuracy:', accuracy_maj, 'Precision:', precision_maj, 'Recall:',recall_maj, 'F1-Score:',f1_maj)