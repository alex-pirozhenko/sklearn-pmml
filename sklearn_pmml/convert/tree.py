from functools import partial

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree._tree import Tree, TREE_LEAF
import numpy as np

from sklearn_pmml.convert import EstimatorConverter
from sklearn_pmml.convert.features import Feature, CategoricalFeature, NumericFeature
import sklearn_pmml.pmml as pmml
from sklearn_pmml.convert import estimator_to_converter


class DecisionTreeConverter(EstimatorConverter):
    SPLIT_BINARY = 'binarySplit'
    OPERATOR_LE = 'lessOrEqual'
    NODE_ROOT = 0

    def __init__(self, estimator, context, mode):
        super(DecisionTreeConverter, self).__init__(estimator, context, mode)

        assert len(self.context.schemas[self.SCHEMA_OUTPUT]) == 1, 'Only one-label trees are supported'
        assert hasattr(estimator, 'tree_'), 'Estimator has no tree_ attribute'
        if mode == self.MODE_CLASSIFICATION:
            assert isinstance(self.context.schemas[self.SCHEMA_OUTPUT][0], CategoricalFeature), \
                'Only a categorical feature can be an output of classification'
            assert isinstance(self.estimator, ClassifierMixin), \
                'Only a classifier can be serialized in classification mode'
        if mode == self.MODE_REGRESSION:
            assert isinstance(self.context.schemas[self.SCHEMA_OUTPUT][0], NumericFeature), \
                'Only a numeric feature can be an output of regression'
            assert isinstance(self.estimator, RegressorMixin), \
                'Only a regressor can be serialized in regression mode'
        assert estimator.tree_.value.shape[1] == len(self.context.schemas[self.SCHEMA_OUTPUT]), \
            'Tree outputs {} results while the schema specifies {} output fields'.format(
                estimator.tree_.value.shape[1], len(self.context.schemas[self.SCHEMA_OUTPUT]))

    def model(self, verification_data=None):
        assert self.SCHEMA_NUMERIC in self.context.schemas, \
            'Either build transformation dictionary or provide {} schema in context'.format(self.SCHEMA_NUMERIC)
        tm = pmml.TreeModel(functionName=self.model_function_name, splitCharacteristic=self.SPLIT_BINARY)
        tm.append(self.mining_schema())
        tm.Node = self._transform_node(
            self.estimator.tree_,
            self.NODE_ROOT,
            self.context.schemas[self.SCHEMA_NUMERIC],
            self.context.schemas[self.SCHEMA_OUTPUT][0]
        )
        if verification_data is not None:
            tm.append(self.model_verification(verification_data))
        return tm

    def _transform_node(self, tree, index, input_schema, output_feature, enter_condition=None):
        """
        Recursive mapping of sklearn Tree into PMML Node tree
        :return: Node element
        """
        assert isinstance(tree, Tree)
        assert isinstance(input_schema, list)
        assert isinstance(output_feature, Feature)

        node = pmml.Node()
        if enter_condition is None:
            node.append(pmml.True_())
        else:
            node.append(enter_condition)
        node.recordCount = tree.n_node_samples[index]

        if tree.children_left[index] != TREE_LEAF:
            feature = input_schema[tree.feature[index]]
            assert isinstance(feature, Feature)
            left_child = self._transform_node(
                tree,
                tree.children_left[index],
                input_schema,
                output_feature,
                enter_condition=pmml.SimplePredicate(
                    field=feature.full_name, operator=DecisionTreeConverter.OPERATOR_LE, value_=tree.threshold[index]
                )
            )
            right_child = self._transform_node(tree, tree.children_right[index], input_schema, output_feature)
            if self.model_function_name == self.MODE_CLASSIFICATION:
                score, score_prob = None, 0.0
                for i in range(len(tree.value[index][0])):
                    left_score = left_child.ScoreDistribution[i]
                    right_score = right_child.ScoreDistribution[i]
                    prob = float(left_score.recordCount + right_score.recordCount) / node.recordCount
                    node.append(pmml.ScoreDistribution(
                        recordCount=left_score.recordCount + right_score.recordCount,
                        value_=left_score.value_,
                        confidence=prob
                    ))
                    if score_prob < prob:
                        score, score_prob = left_score.value_, prob
                node.score = score
            node.append(left_child).append(right_child)

        else:
            node_value = np.array(tree.value[index][0])
            if self.model_function_name == self.MODE_CLASSIFICATION:
                probs = node_value / float(node_value.sum())
                for i in range(len(probs)):
                    node.append(pmml.ScoreDistribution(
                        confidence=probs[i],
                        recordCount=node_value[i],
                        value_=output_feature.from_number(i)
                    ))
                node.score = output_feature.from_number(probs.argmax())
            elif self.model_function_name == self.MODE_REGRESSION:
                node.score = node_value[0]

        return node


estimator_to_converter[DecisionTreeClassifier] = partial(
    DecisionTreeConverter, mode=DecisionTreeConverter.MODE_CLASSIFICATION
)
estimator_to_converter[DecisionTreeRegressor] = partial(
    DecisionTreeConverter, mode=DecisionTreeConverter.MODE_REGRESSION
)