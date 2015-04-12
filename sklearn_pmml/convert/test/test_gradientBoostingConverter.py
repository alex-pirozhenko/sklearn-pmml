from unittest import TestCase
from sklearn.ensemble import GradientBoostingClassifier
from sklearn_pmml import PMMLTransformationContext, PMMLBuilder
from sklearn_pmml.convert import Schema
from sklearn_pmml.convert.features import *
from sklearn_pmml.convert.gbrt import GradientBoostingConverter


class TestGradientBoostingClassifierConverter(TestCase):

    def setUp(self):
        super(TestGradientBoostingClassifierConverter, self).setUp()
        self.est = GradientBoostingClassifier(max_depth=2, n_estimators=1)
        self.est.fit([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ], [0, 1, 1, 1])
        self.converter = GradientBoostingConverter(
            mode=GradientBoostingConverter.MODE_CLASSIFICATION
        )
        self.ctx = PMMLTransformationContext(
            schema=Schema(features=[IntegerNumericFeature('x1'), StringCategoricalFeature('x2', ['zero', 'one']), ],
                          output=StringCategoricalFeature('output', ['neg', 'pos'])), metadata={})

    def test_transform(self):
        a = list(self.converter.transform(self.est, self.ctx))
        # TODO: add real tests

    def test_end_to_end(self):
        print self.est.predict_proba([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ])
        print PMMLBuilder().build(self.est, self.ctx).toDOM().toprettyxml()

