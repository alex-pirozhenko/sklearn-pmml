from unittest import TestCase
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn_pmml import PMMLTransformationContext
from sklearn_pmml.convert import Schema
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
        self.converter = DecisionTreeConverter(
            mode=DecisionTreeConverter.MODE_CLASSIFICATION
        )
        self.ctx = PMMLTransformationContext(
            schema=Schema(features=[IntegerNumericFeature('x1'), StringCategoricalFeature('x2', ['zero', 'one']), ],
                          output=StringCategoricalFeature('output', ['neg', 'pos'])), metadata={})

    def test_is_applicable(self):
        assert self.converter.is_applicable(self.est, self.ctx)

    def test_transform(self):
        tm = self.converter.transform(self.est, self.ctx)
        assert tm.MiningSchema is not None, 'Missing mining schema'
        assert len(tm.MiningSchema.MiningField) == 3, 'Wrong number of mining fields'
        assert tm.Node is not None, 'Missing root node'
        assert tm.Node.recordCount == 4
        assert tm.Node.score == 'pos'
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
        self.converter = DecisionTreeConverter(
            mode=DecisionTreeConverter.MODE_REGRESSION
        )
        self.ctx = PMMLTransformationContext(
            schema=Schema(features=[IntegerNumericFeature('x1'), StringCategoricalFeature('x2', ['zero', 'one']), ],
                          output=IntegerNumericFeature('output')), metadata={})

    def test_is_applicable(self):
        assert self.converter.is_applicable(self.est, self.ctx)

    def test_transform(self):
        tm = self.converter.transform(self.est, self.ctx)
        assert tm.MiningSchema is not None, 'Missing mining schema'
        assert len(tm.MiningSchema.MiningField) == 3, 'Wrong number of mining fields'
        assert tm.Node is not None, 'Missing root node'
        assert tm.Node.recordCount == 4
        assert tm.Node.True_ is not None, 'Root condition should always be True'