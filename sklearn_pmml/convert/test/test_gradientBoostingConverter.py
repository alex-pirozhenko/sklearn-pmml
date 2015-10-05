from unittest import TestCase

from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

from sklearn_pmml.convert.test.jpmml_test import JPMMLClassificationTest, JPMMLTest, TARGET_NAME
from sklearn_pmml.convert import TransformationContext
from sklearn_pmml.convert.features import *
from sklearn_pmml.convert.gbrt import GradientBoostingConverter


class TestGradientBoostingClassifierConverter(TestCase):
    def setUp(self):
        np.random.seed(1)
        self.est = GradientBoostingClassifier(max_depth=2, n_estimators=10)
        self.est.fit([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ], [0, 1, 1, 1])
        self.ctx = TransformationContext(
            input=[IntegerNumericFeature('x1'), StringCategoricalFeature('x2', ['zero', 'one'])],
            derived=[],
            model=[IntegerNumericFeature('x1'), StringCategoricalFeature('x2', ['zero', 'one'])],
            output=[RealNumericFeature('output')]
        )
        self.converter = GradientBoostingConverter(
            estimator=self.est,
            context=self.ctx
        )

    def test_transform(self):
        p = self.converter.pmml()
        mm = p.MiningModel[0]
        assert mm.MiningSchema is not None, 'Missing mining schema'
        assert len(mm.MiningSchema.MiningField) == 3, 'Wrong number of mining fields'
        assert mm.Segmentation is not None, 'Missing segmentation root'

    def test_transform_with_verification(self):
        p = self.converter.pmml([
            {'x1': 0, 'x2': 'zero', 'output': self.est.predict_proba([[0, 0]])[0, 1]},
            {'x1': 0, 'x2': 'one', 'output': self.est.predict_proba([[0, 1]])[0, 1]},
            {'x1': 1, 'x2': 'zero', 'output': self.est.predict_proba([[1, 0]])[0, 1]},
            {'x1': 1, 'x2': 'one', 'output': self.est.predict_proba([[1, 1]])[0, 1]},
        ])
        mm = p.MiningModel[0]
        assert mm.MiningSchema is not None, 'Missing mining schema'
        assert len(mm.MiningSchema.MiningField) == 3, 'Wrong number of mining fields'
        assert mm.Segmentation is not None, 'Missing segmentation root'


class TestGradientBoostingClassifierParity(TestCase, JPMMLClassificationTest):

    @classmethod
    def setUpClass(cls):
        if JPMMLTest.can_run():
            JPMMLTest.init_jpmml()

    def setUp(self):
        self.model = GradientBoostingClassifier(n_estimators=2)
        self.init_data_one_label()
        self.converter = GradientBoostingConverter(
            estimator=self.model,
            context=self.ctx
        )
