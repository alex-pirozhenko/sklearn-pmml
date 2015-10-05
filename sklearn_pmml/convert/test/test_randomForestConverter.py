from sklearn_pmml.convert import IntegerCategoricalFeature
from sklearn_pmml.convert.test.jpmml_test import JPMMLClassificationTest, JPMMLTest, TARGET_NAME
from unittest import TestCase
from sklearn.ensemble import RandomForestClassifier

__author__ = 'evancox'


from sklearn_pmml.convert.random_forest import RandomForestClassifierConverter


class TestRandomForestClassifierParity(TestCase, JPMMLClassificationTest):

    @classmethod
    def setUpClass(cls):
        if JPMMLTest.can_run():
            JPMMLTest.init_jpmml()

    def setUp(self):
        self.model = RandomForestClassifier(
            n_estimators=3,
            max_depth=3
        )
        self.init_data()
        self.converter = RandomForestClassifierConverter(
            estimator=self.model,
            context=self.ctx
        )

    @property
    def output(self):
        return IntegerCategoricalFeature(name=TARGET_NAME, value_list=[0, 1, 2])
