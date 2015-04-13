from unittest import TestCase

from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

from sklearn_pmml.convert import TransformationContext
from sklearn_pmml.convert.features import *
from sklearn_pmml.convert.gbrt import GradientBoostingConverter


class TestGradientBoostingClassifierConverter(TestCase):
    def setUp(self):
        np.random.seed(1)
        self.est = GradientBoostingClassifier(max_depth=2, n_estimators=1)
        self.est.fit([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ], [0, 1, 1, 1])
        self.ctx = TransformationContext(
            input=[IntegerNumericFeature('x1'), StringCategoricalFeature('x2', ['zero', 'one'])],
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

