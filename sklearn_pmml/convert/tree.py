from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree._tree import Tree, TREE_LEAF
from sklearn_pmml.convert import Converter, PMMLTransformationContext, Schema
from sklearn_pmml.convert.features import Feature, CategoricalFeature, NumericFeature
import sklearn_pmml.pmml as pmml
import numpy as np
from sklearn_pmml.convert import estimator_to_converter


class DecisionTreeConverter(Converter):

    BINARY_SPLIT = 'binarySplit'

    OPERATOR_LE = 'lessOrEqual'

    def __init__(self, mode):
        assert (mode is None) or (mode in self.all_modes)
        self.mode = mode

    def is_applicable(self, obj, ctx):
        if not hasattr(obj, 'tree_'):
            return False
        if self.mode == self.MODE_CLASSIFICATION:
            if not isinstance(ctx.schema.output, CategoricalFeature):
                return False
            if not isinstance(obj, ClassifierMixin):
                return False
        if self.mode == self.MODE_REGRESSION:
            if not isinstance(ctx.schema.output, NumericFeature):
                return False
            if not isinstance(obj, RegressorMixin):
                return False

        return True

    def transform(self, obj, ctx):
        super(DecisionTreeConverter, self).transform(obj, ctx)
        assert isinstance(ctx, PMMLTransformationContext)
        tm = pmml.TreeModel(functionName=self.mode, splitCharacteristic=self.BINARY_SPLIT)
        tm.append(self.mining_schema(ctx.schema))
        tm.Node = self._transform_node(obj.tree_, 0, ctx.schema)
        yield tm

    def _transform_node(self, tree, index, schema, enter_condition=None):
        assert isinstance(tree, Tree)
        assert isinstance(schema, Schema)
        output_feature = schema.output

        if self.mode == self.MODE_CLASSIFICATION:
            assert isinstance(output_feature, CategoricalFeature)
        elif self.mode == self.MODE_REGRESSION:
            assert isinstance(output_feature, NumericFeature)

        node = pmml.Node()
        if enter_condition is None:
            node.append(pmml.True_())
        else:
            node.append(enter_condition)
        node.recordCount = tree.n_node_samples[index]

        if tree.children_left[index] != TREE_LEAF:
            feature = schema.features[tree.feature[index]]
            assert isinstance(feature, Feature)
            left_child = self._transform_node(
                tree, tree.children_left[index], schema, enter_condition=(pmml.SimplePredicate(
                    field=feature.internal_name,
                    operator=DecisionTreeConverter.OPERATOR_LE,
                    value_=tree.threshold[index]
                ))
            )
            right_child = self._transform_node(
                tree, tree.children_right[index], schema
            )
            if self.mode == self.MODE_CLASSIFICATION:
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
            assert len(tree.value[index]) == 1, 'Only one-label trees are supported'
            node_value = np.array(tree.value[index][0])
            if self.mode == self.MODE_CLASSIFICATION:
                assert node.recordCount == node_value.sum()
                probs = node_value / float(node_value.sum())
                for i in range(len(probs)):
                    node.append(pmml.ScoreDistribution(
                        confidence=probs[i],
                        recordCount=node_value[i],
                        value_=output_feature.from_number(i)
                    ))
                node.score = output_feature.from_number(probs.argmax())
            elif self.mode == self.MODE_REGRESSION:
                node.score = node_value[0]

        return node


estimator_to_converter[DecisionTreeClassifier] = DecisionTreeConverter(mode=DecisionTreeConverter.MODE_CLASSIFICATION)
estimator_to_converter[DecisionTreeRegressor] = DecisionTreeConverter(mode=DecisionTreeConverter.MODE_REGRESSION)