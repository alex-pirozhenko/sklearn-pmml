from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import Tree, TREE_LEAF
from sklearn_pmml.convert import Converter, PMMLTransformationContext, Schema
from sklearn_pmml.convert.features import Feature, CategoricalFeature
import sklearn_pmml.pmml as pmml
import numpy as np
from sklearn_pmml.convert import estimator_to_converter


class DecisionTreeClassifierConverter(Converter):
    MODE_CLASSIFICATION = 'classification'
    # MODE_REGRESSION = 'regression'
    __all_modes = {
        MODE_CLASSIFICATION,
        # MODE_REGRESSION
    }

    BINARY_SPLIT = 'binarySplit'

    OPERATOR_LE = 'lessOrEqual'

    def __init__(self, mode):
        assert (mode is None) or (mode in self.__all_modes)
        self.mode = mode

    def is_applicable(self, obj, ctx):
        return hasattr(obj, 'tree_')

    def transform(self, obj, ctx):
        super(DecisionTreeClassifierConverter, self).transform(obj, ctx)
        assert isinstance(ctx, PMMLTransformationContext)
        assert isinstance(obj, DecisionTreeClassifier)
        tm = pmml.TreeModel(functionName=self.mode, splitCharacteristic=self.BINARY_SPLIT)
        tm.append(self.mining_schema(ctx.schema))
        tm.Node = self._transform_node(obj.tree_, 0, ctx.schema)
        return tm

    @staticmethod
    def _transform_node(tree, index, schema, enter_condition=None):
        assert isinstance(tree, Tree)
        assert isinstance(schema, Schema)
        output_feautre = schema.output
        assert isinstance(output_feautre, CategoricalFeature)
        node = pmml.Node()
        if enter_condition is None:
            node.append(pmml.True_())
        else:
            node.append(enter_condition)
        node.recordCount = tree.n_node_samples[index]

        if tree.children_left[index] != TREE_LEAF:
            feature = schema.features[tree.feature[index]]
            assert isinstance(feature, Feature)
            left_child = DecisionTreeClassifierConverter._transform_node(
                tree, tree.children_left[index], schema, enter_condition=(pmml.SimplePredicate(
                    field=feature.internal_name,
                    operator=DecisionTreeClassifierConverter.OPERATOR_LE,
                    value_=tree.threshold[index]
                ))
            )
            right_child = DecisionTreeClassifierConverter._transform_node(
                tree, tree.children_right[index], schema
            )
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
            value_counts = np.array(tree.value[index][0])
            assert node.recordCount == value_counts.sum()
            probs = value_counts / float(value_counts.sum())
            for i in range(len(probs)):
                node.append(pmml.ScoreDistribution(
                    confidence=probs[i],
                    recordCount=value_counts[i],
                    value_=output_feautre.from_number(i)
                ))
            node.score = output_feautre.from_number(probs.argmax())

        return node


estimator_to_converter[DecisionTreeClassifier] = DecisionTreeClassifierConverter(mode=DecisionTreeClassifierConverter.MODE_CLASSIFICATION)