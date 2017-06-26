# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#plt.style.use('ggplot')

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'
data = pd.read_csv(url, header=None, na_values='?')

data.columns = ['A' + str(i) for i in range(1, 16)] + ['class']
#print (data.head())
#print(data.describe())

#--------------- ВЫДЕЛЯЕМ ЧИСЛОВЫЕ И НЕ ЧИСЛОВЫЕ ДАННЫЕ
categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']
numerical_columns   = [c for c in data.columns if data[c].dtype.name != 'object']

#print (categorical_columns)
#print (numerical_columns)

#--------------- Описание категорий
#print(data[categorical_columns].describe())

#for c in categorical_columns:
#    print (data[c].unique())
    
#from pandas.tools.plotting import scatter_matrix
#scatter_matrix(data, alpha=0.05, figsize=(10, 10));

#--------------- ТАБЛИЦА КОРРЕЛЯЦИЙ
#print (data.corr())

#col1 = 'A2'
#col2 = 'A3'

#plt.figure(figsize=(10, 6))

#plt.scatter(data[col1][data['class'] == '+'],
#            data[col2][data['class'] == '+'],
#            alpha=0.75,
#            color='red',
#            label='+')

#plt.scatter(data[col1][data['class'] == '-'],
#            data[col2][data['class'] == '-'],
#            alpha=0.75,
#            color='blue',
#            label='-')

#plt.xlabel(col1)
#plt.ylabel(col2)
#plt.legend(loc='best');

#--------------- Заполняем средними значениями пустые ячейки
data = data.fillna(data.median(axis=0), axis=0)

#--------------- Заполняем пустые ячейки самыми часто втречающимися.!!! НЕВЕРНО !!! НУЖНО СОХРАНЯТЬ ПРОПОРЦИИ
data_describe = data.describe(include=[object])
for c in categorical_columns:
    data[c] = data[c].fillna(data_describe[c]['top'])

#print(data.describe(include=[object]))

#--------------- Не числовые данные приводим к числовым
binary_columns    = [c for c in categorical_columns if data_describe[c]['unique'] == 2]
nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]
#print (binary_columns, nonbinary_columns)

for c in binary_columns:
    top = data_describe[c]['top']
    top_items = data[c] == top
    data.loc[top_items, c] = 0
    data.loc[np.logical_not(top_items), c] = 1
#print(data[binary_columns].describe())
data_nonbinary = pd.get_dummies(data[nonbinary_columns])
#print (data_nonbinary.columns)

#--------------- Нормализуем числовые данные
data_numerical = data[numerical_columns]
data_numerical = (data_numerical - data_numerical.mean()) / data_numerical.std()
#print(data[numerical_columns].describe())

#--------------- Соединяем все в одну таблицу
data = pd.concat((data_numerical, data[binary_columns], data_nonbinary), axis=1)
data = pd.DataFrame(data, dtype=float)
#print (data.shape)
#print (data.columns)
#print (data.describe())
X = data.drop(('class'), axis=1)  # Выбрасываем столбец 'class'.
y = data['class']
feature_names = X.columns
#print (feature_names)
#print (X.shape)
#print (y.shape)
N, d = X.shape

#--------------- Разбиваем выборку на тестовую и тренировочную
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 11)

N_train, _ = X_train.shape 
N_test,  _ = X_test.shape 
#print (N_train, N_test)



'''
#--------------- !!!!! kNN – метод ближайших соседей !!!!!

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_train_predict = knn.predict(X_train)
y_test_predict = knn.predict(X_test)

err_train = np.mean(y_train != y_train_predict)
err_test  = np.mean(y_test  != y_test_predict)
#print (err_train, err_test)

#--------------- Поиск оптимальных значений параметров
from sklearn.grid_search import GridSearchCV
n_neighbors_array = [1, 3, 5, 7, 10, 15]
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid={'n_neighbors': n_neighbors_array})
grid.fit(X_train, y_train)

best_cv_err = 1 - grid.best_score_
best_n_neighbors = grid.best_estimator_.n_neighbors
#print (best_cv_err, best_n_neighbors)
#--------------- В качестве оптимального метод выбрал значение kk равное 7.
knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
knn.fit(X_train, y_train)

err_train = np.mean(y_train != knn.predict(X_train))
err_test  = np.mean(y_test  != knn.predict(X_test))
print ('KNN - ', err_train, err_test)


#--------------- !!!!! SVC – машина опорных векторов !!!!!

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)

err_train = np.mean(y_train != svc.predict(X_train))
err_test  = np.mean(y_test  != svc.predict(X_test))
#print (err_train, err_test)
#--------------- Вначале попробуем найти лучшие значения параметров для радиального ядра.
from sklearn.grid_search import GridSearchCV
C_array = np.logspace(-3, 3, num=7)
gamma_array = np.logspace(-5, 2, num=8)
svc = SVC(kernel='rbf')
grid = GridSearchCV(svc, param_grid={'C': C_array, 'gamma': gamma_array})
grid.fit(X_train, y_train)
#print ('CV error    = ', 1 - grid.best_score_)
#print ('best C      = ', grid.best_estimator_.C)
#print ('best gamma  = ', grid.best_estimator_.gamma)

#CV error    =  0.138716356108
#best C      =  1.0
#best gamma  =  0.01

svc = SVC(kernel='rbf', C=grid.best_estimator_.C, gamma=grid.best_estimator_.gamma)
svc.fit(X_train, y_train)

err_train = np.mean(y_train != svc.predict(X_train))
err_test  = np.mean(y_test  != svc.predict(X_test))
print ('SVC radial - ', err_train, err_test)



#--------------- !!!!! Линейное ядро !!!!!
#--------------- Находим лучшие значения параметров для линейного ядра.
from sklearn.grid_search import GridSearchCV
C_array = np.logspace(-3, 3, num=7)
svc = SVC(kernel='linear')
grid = GridSearchCV(svc, param_grid={'C': C_array})
grid.fit(X_train, y_train)
#print ('CV error    = ', 1 - grid.best_score_)
#print ('best C      = ', grid.best_estimator_.C)
#--------------- Проверяем результат
svc = SVC(kernel='linear', C=grid.best_estimator_.C)
svc.fit(X_train, y_train)
err_train = np.mean(y_train != svc.predict(X_train))
err_test  = np.mean(y_test  != svc.predict(X_test))
print ('SVC linear - ', err_train, err_test)



#--------------- !!!!! Полиномиальное ядро !!!!!
#--------------- Находим лучшие значения параметров для полиномиального ядра.
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
C_array = np.logspace(-5, 2, num=8)
gamma_array = np.logspace(-5, 2, num=8)
degree_array = [2, 3, 4]
svc = SVC(kernel='poly')
grid = GridSearchCV(svc, param_grid={'C': C_array, 'gamma': gamma_array, 'degree': degree_array})
grid.fit(X_train, y_train)
#print ('CV error    = ', 1 - grid.best_score_)
#print ('best C      = ', grid.best_estimator_.C)
#print ('best gamma  = ', grid.best_estimator_.gamma)
#print ('best degree = ', grid.best_estimator_.degree)

svc = SVC(kernel='poly', C=grid.best_estimator_.C, 
          gamma=grid.best_estimator_.gamma, degree=grid.best_estimator_.degree)
svc.fit(X_train, y_train)

err_train = np.mean(y_train != svc.predict(X_train))
err_test  = np.mean(y_test  != svc.predict(X_test))
print (err_train, err_test)
'''



#--------------- !!!!! Random Forest – случайный лес !!!!!
from sklearn import ensemble
rf = ensemble.RandomForestClassifier(n_estimators=100, random_state=11)
rf.fit(X_train, y_train)

err_train = np.mean(y_train != rf.predict(X_train))
err_test  = np.mean(y_test  != rf.predict(X_test))
print (err_train, err_test)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]


#--------------- !!!!! Выделяем самые коррелирующие признаки !!!!!
print("Feature importances:")
for f, idx in enumerate(indices):
    print("{:2d}. feature '{:5s}' ({:.4f})".format(f + 1, feature_names[idx], importances[idx]))
best_features = indices[:8]
best_features_names = feature_names[best_features]
print(best_features_names)




#--------------- !!!!! Градиентный бустинг !!!!!
from sklearn import ensemble
gbt = ensemble.GradientBoostingClassifier(n_estimators=100, random_state=11)
gbt.fit(X_train, y_train)

err_train = np.mean(y_train != gbt.predict(X_train))
err_test = np.mean(y_test != gbt.predict(X_test))
print (err_train, err_test)

gbt = ensemble.GradientBoostingClassifier(n_estimators=100, random_state=11)
gbt.fit(X_train[best_features_names], y_train)

err_train = np.mean(y_train != gbt.predict(X_train[best_features_names]))
err_test = np.mean(y_test != gbt.predict(X_test[best_features_names]))
print (err_train, err_test)