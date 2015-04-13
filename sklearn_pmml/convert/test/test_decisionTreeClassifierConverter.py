from unittest import TestCase

import numpy as np

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn_pmml.convert import TransformationContext
from sklearn_pmml.convert.features import *
from sklearn_pmml.convert.tree import DecisionTreeConverter


class TestDecisionTreeClassifierConverter(TestCase):
    def setUp(self):
        np.random.seed(1)
        self.est = DecisionTreeClassifier(max_depth=2)
        self.est.fit([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ], [0, 1, 1, 1])
        self.ctx = TransformationContext(
            input=[IntegerNumericFeature('x1'), StringCategoricalFeature('x2', ['zero', 'one'])],
            output=[IntegerCategoricalFeature('output', ['neg', 'pos'])]
        )
        self.converter = DecisionTreeConverter(
            estimator=self.est,
            context=self.ctx,
            mode=DecisionTreeConverter.MODE_CLASSIFICATION
        )

    def test_transform(self):
        p = self.converter.pmml()
        tm = p.TreeModel[0]
        assert tm.MiningSchema is not None, 'Missing mining schema'
        assert len(tm.MiningSchema.MiningField) == 3, 'Wrong number of mining fields'
        assert tm.Node is not None, 'Missing root node'
        assert tm.Node.recordCount == 4
        assert tm.Node.True_ is not None, 'Root condition should always be True'


class TestDecisionTreeRegressorConverter(TestCase):
    def setUp(self):
        np.random.seed(1)
        self.est = DecisionTreeRegressor(max_depth=2)
        self.est.fit([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ], [0, 1, 1, 1])
        self.ctx = TransformationContext(
            input=[IntegerNumericFeature('x1'), StringCategoricalFeature('x2', ['zero', 'one'])],
            output=[IntegerNumericFeature('output')]
        )
        self.converter = DecisionTreeConverter(
            estimator=self.est,
            context=self.ctx,
            mode=DecisionTreeConverter.MODE_REGRESSION
        )

    def test_transform(self):
        p = self.converter.pmml()
        tm = p.TreeModel[0]
        assert tm.MiningSchema is not None, 'Missing mining schema'
        assert len(tm.MiningSchema.MiningField) == 3, 'Wrong number of mining fields'
        assert tm.Node is not None, 'Missing root node'
        assert tm.Node.recordCount == 4
        assert tm.Node.True_ is not None, 'Root condition should always be True'