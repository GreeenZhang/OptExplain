import numpy as np
import math


class Extractor(object):
    def __init__(self, estimators=None, phi=0, theta=0, psi=0):
        # phi:rule阈值    theta:node阈值    psi:签名取模的参数
        self._estimators = estimators
        self._forest_paths = []         # 存森林的所有路径
        self.scale = 0
        self._forest_formulae = []      # 存filter后的所有路径；元素shape(2, n_feature)，第一行上界，第二行下界
        self._forest_formulae_visited = []
        self._forest_values = []        # 存filter后的所有路径的叶子值
        self._forest_weights = []       # 存filter后的所有路径的权重
        self._forest_signatures = []    # 存filter后的所有路径的签名
        self.n_estimators = estimators.n_estimators
        self.n_classes = estimators.n_classes_
        self.n_features = estimators.n_features_
        self.n_outputs = estimators.n_outputs_
        self._phi = phi
        self._theta = theta
        self._psi = psi
        self.n_original_leaves_num = 0

        self._quality = []
        self._ig = []
        self.max_rule = 0   # rule质量最好的quality值
        self.max_node = 0   # node信息增益最高的值
        self.min_rule = 1
        self.min_node = 1

    def set_param(self, phi, theta, psi):
        self._phi = phi
        self._theta = theta
        self._psi = psi

    def opt_set_quality(self, quality, ig):
        self._quality = quality
        self._ig = ig

    def opt_get_quality(self):
        return self._quality, self._ig

    def opt_clear_quality(self):
        self._quality = []
        self._ig = []

    def extract_tree_paths(self, estimator):
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        # print(children_left)
        # print(children_right)

        result = list()
        stack = [(0, [])]
        while len(stack):
            # print(stack)
            node, p = stack.pop()
            if children_right[node] == children_left[node]:
                p.append(node)
                result.append(p[:])
                # print(result)
            else:
                p.append(node)
                if children_right[node] != -1:
                    stack.append((children_right[node], p[:]))
                if children_left[node] != -1:
                    stack.append((children_left[node], p[:]))

        return result

    def extract_forest_paths(self):
        for estimator in self._estimators:
            paths = self.extract_tree_paths(estimator)
            for path in paths:
                self.scale += (len(path)-1)
            self._forest_paths.append(paths)

    def count_quality(self):
        res_quality = []
        _n_classes = self._estimators.n_classes_
        for index, tree in enumerate(self._forest_paths):
            acc = self._estimators.trees_oob_scores[index]
            _tree = self._estimators[index].tree_
            rule_quality = []
            node_quality = []
            for rule in tree:
                _node_quality = []
                self.n_original_leaves_num += 1

                quality = (1-(_tree.impurity[rule[-1]]/math.log(_n_classes, 2)))*acc  # rule质量公式
                rule_quality.append(quality)

                if quality > self.max_rule:
                    self.max_rule = quality
                if quality < self.min_rule:
                    self.min_rule = quality

                for node in rule[:-1]:
                    father_value = _tree.value[node].sum()
                    l_value = _tree.value[_tree.children_left[node]].sum()
                    r_value = _tree.value[_tree.children_right[node]].sum()
                    l_entropy = _tree.impurity[_tree.children_left[node]]
                    r_entropy = _tree.impurity[_tree.children_right[node]]
                    ig = _tree.impurity[
                             node] - l_value / father_value * l_entropy - r_value / father_value * r_entropy  # 节点的信息增益

                    _node_quality.append(ig)
                    if ig > self.max_node:
                        self.max_node = ig
                    if ig < self.min_node:
                        self.min_node = ig
                node_quality.append(_node_quality)
            self._quality.append(rule_quality)
            self._ig.append(node_quality)


    def rule_filter(self):
        if len(self._quality) == 0:
            self.count_quality()
            # print('count')

        if self._phi > self.max_rule or self._theta > self.max_node:
            return False

        for index, tree in enumerate(self._forest_paths):
            for j, rule in enumerate(tree):
                quality = self._quality[index][j]
                if quality >= self._phi:
                    if self.node_filter(j, index) == 1:
                        signature = np.array(self._estimators[index].tree_.value[rule[-1]][0])
                        leaf_sum = self._estimators[index].tree_.value[rule[-1]].sum()
                        signature = np.ceil(signature / leaf_sum / self._psi)
                        self._forest_signatures.append(tuple(signature))    # 存签名
                        self._forest_weights.append(quality)  # 存rule权重

        return True

        # _n_classes = self._estimators.n_classes_
        # for index, tree in enumerate(self._forest_paths):
        #     acc = self._estimators.trees_oob_scores[index]
        #     for rule in tree:
        #         self.n_original_leaves_num += 1
        #         quality = (1-(self._estimators[index].tree_.impurity[rule[-1]]/math.log(_n_classes, 2)))*acc  # rule质量公式
        #
        #         if quality > self.max_rule:
        #             self.max_rule = quality
        #         if quality < self.min_rule:
        #             self.min_rule = quality
        #
        #         if quality >= self._phi:
        #             if self.node_filter(rule, index) == 1:
        #                 signature = np.array(self._estimators[index].tree_.value[rule[-1]][0])
        #                 leaf_sum = self._estimators[index].tree_.value[rule[-1]].sum()
        #                 signature = np.ceil(signature / leaf_sum / self._psi)
        #                 self._forest_signatures.append(tuple(signature))    # 存签名
        #                 self._forest_weights.append(quality)  # 存rule权重

            # for rule in tree:
            #     self.node_filter(rule, index)

    def node_filter(self, j, index):
        _tree = self._estimators[index].tree_
        _feature = _tree.feature
        _threshold = _tree.threshold
        formula = np.zeros([2, self.n_features], dtype=float)    # 记录该rule的formula
        visited = np.zeros([2, self.n_features], dtype=float)    # formula的标志数组


        # for k, node in enumerate(rule[:-1]):
        #     father_value = _tree.value[node].sum()
        #     l_value = _tree.value[_tree.children_left[node]].sum()
        #     r_value = _tree.value[_tree.children_right[node]].sum()
        #     l_entropy = _tree.impurity[_tree.children_left[node]]
        #     r_entropy = _tree.impurity[_tree.children_right[node]]
        #     ig = _tree.impurity[node] - l_value/father_value*l_entropy - r_value/father_value*r_entropy     # 节点的信息增益
        #
        #     if ig > self.max_node:
        #         self.max_node = ig
        #     if ig < self.min_node:
        #         self.min_node = ig
        #
        #     if ig >= self._theta:  # 节点过滤 ig >= theta
        #         if _tree.children_left[node] == rule[k+1]:
        #             if not visited[0, _feature[node]]:
        #                 visited[0, _feature[node]] = 1
        #                 formula[0, _feature[node]] = _threshold[node]
        #             else:
        #                 if formula[0, _feature[node]] > _threshold[node]:
        #                     formula[0, _feature[node]] = _threshold[node]
        #         else:
        #             if not visited[1, _feature[node]]:
        #                 visited[1, _feature[node]] = 1
        #                 formula[1, _feature[node]] = _threshold[node]
        #             else:
        #                 if formula[1, _feature[node]] < _threshold[node]:
        #                     formula[1, _feature[node]] = _threshold[node]


        rule = self._forest_paths[index][j]
        for k, node in enumerate(rule[:-1]):
            ig = self._ig[index][j][k]

            if ig >= self._theta:       # 节点过滤 ig >= theta
                if _tree.children_left[node] == rule[k+1]:
                    if not visited[0, _feature[node]]:
                        visited[0, _feature[node]] = 1
                        formula[0, _feature[node]] = _threshold[node]
                    else:
                        if formula[0, _feature[node]] > _threshold[node]:
                            formula[0, _feature[node]] = _threshold[node]
                else:
                    if not visited[1, _feature[node]]:
                        visited[1, _feature[node]] = 1
                        formula[1, _feature[node]] = _threshold[node]
                    else:
                        if formula[1, _feature[node]] < _threshold[node]:
                            formula[1, _feature[node]] = _threshold[node]


        if not np.all(visited == 0):                # 若rule中还存在非叶子节点，才存储rule
            self._forest_values.append(_tree.value[rule[-1]][0])  # 存叶子
            self._forest_formulae.append(formula)  # 存公式
            self._forest_formulae_visited.append(visited)
            return 1
        else:
            return 0

    @property
    def forest_values(self):
        return self._forest_values


