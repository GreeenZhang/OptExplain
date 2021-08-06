import sys
import os
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from Main_Process import MainProcess
from Extractor import Extractor
from Z3Process import Z3Process
from FormulaeEstimator import FormulaeEstimator

if __name__ == "__main__":
    model_name = 'diabetes'
    file_name = './datasets/'+model_name+'.csv'
    model = pd.read_csv(file_name)
    X = model.iloc[:, 0:-1]
    y = model.iloc[:, -1]

    train_samples = 668

    # 划分训练集 和测试集
    X_train, X_test, y_train, y_test = train_test_split(
             X, y, train_size=train_samples, test_size=10, random_state=0)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print('RF...')

    n_estimators = 100
    max_depth = 10


    clf = RandomForestClassifier(n_estimators=n_estimators, criterion='entropy', oob_score=True, random_state=0). \
        fit(X_train, y_train)

    file_num = 1
    while os.path.exists('./proClass/' + model_name + str(file_num) + '_proClass.txt') is True:
        file_num += 1
    file = open('./proClass/' + model_name + str(file_num) + '_proClass.txt', 'w')

    m = MainProcess(clf, X_test, y_test, file, generation=20, scale=20,
                    conjunction=False, maxsat_on=True, tailor=False, fitness_func='Pro')
    param = m.pso()
    phi = param[0]
    theta = param[1]
    psi = param[2]
    k = param[3]

    ex = Extractor(clf, phi, theta, psi)
    ex.extract_forest_paths()
    ex.rule_filter()
    print('max_rule', ex.max_rule, 'max_node', ex.max_node)
    print("original path number: ", ex.n_original_leaves_num)
    print('original scale: ', ex.scale)
    print("original path number after rule filter: ", len(ex._forest_values))

    sat = Z3Process(ex, k)
    sat.leaves_partition()
    sat.maxsat()
    sat.run_filter()

    print("original path number after maxsat: ", sat.n_rules_after_max, " after filter: ", sat.n_rules_after_filter, '\n')
    print('classes:', clf.classes_)

    f = FormulaeEstimator(sat, conjunction=True, classes=clf.classes_)
    f.get_formulae_text(file)
    print('scale：', f.scale)
