import sys
import os
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from Main_Process import MainProcess


if __name__ == "__main__":
    # diabetes  credit  creditCard  adult   MiniBooNE   spambase    mnist

    model_name = 'diabetes'
    # 设置训练、优化参数
    n_estimators = 100
    max_depth = 10

    generation = 20
    scale = 20
    acc_weight = 0.5
    conjunction = False
    maxsat_on = False
    size_filter = True

    file_name = './datasets/'+model_name+'.csv'
    model = pd.read_csv(file_name)
    X = model.iloc[:, 0:-1]
    y = model.iloc[:, -1]

    # 划分训练集 和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print('RF...')


    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion='entropy', oob_score=True,
                                 random_state=0). fit(X_train, y_train)
    print(accuracy_score(y_test, clf.predict(X_test)))

    # output
    file_num = 1
    while os.path.exists('./explanation_05/' + model_name + str(file_num) + '.txt') is True:
        file_num += 1
    file = open('./explanation_05/' + model_name + str(file_num) + '.txt', 'w')
    file.write('n_estimators = {}\tmax_depth = {}\n'.format(n_estimators, max_depth))
    file.write('generation = {}\tscale = {}\tacc_weight = {}\tmaxsat = {}\ttailor = {}\n'.
               format(generation, scale, acc_weight, maxsat_on, size_filter))
    print('explain...')
    file.write('begin\n')
    m = MainProcess(clf, X_test, y_test, file, generation=generation, scale=scale, acc_weight=acc_weight,
                    conjunction=conjunction, maxsat_on=maxsat_on, tailor=size_filter, fitness_func='Opt')
    best_param = m.pso()
    # best_param = [0.5, 0.6, 0.5, -1]
    m.explain(best_param, auc_plot=False)
    file.write('end')
    file.close()

