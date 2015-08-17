__author__ = 'evancox'

from jpmml_test import JPMMLClassificationTest, JPMMLTest
from unittest import TestCase

from sklearn.ensemble import RandomForestClassifier


from sklearn_pmml.convert.random_forest import RandomForestClassifierConverter

class TestRandomForestClassifierParity(TestCase, JPMMLClassificationTest):

    @classmethod
    def setUpClass(cls):
        JPMMLTest.init_jpmml()


    def setUp(self):
        self.model = RandomForestClassifier()
        super(TestRandomForestClassifierParity, self).init_data()
        super(TestRandomForestClassifierParity, self).setUp()
        self.converter = RandomForestClassifierConverter(
            estimator=self.model,
            context=self.ctx
        )
