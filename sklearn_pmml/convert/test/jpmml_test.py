__author__ = 'evancox'

import numpy as np
import hashlib
import os
import shutil
import subprocess
import logging

from sklearn_pmml.convert import TransformationContext, Schema
from sklearn_pmml.convert.features import *


TARGET = [0, 1, 2]
TARGET_NAME = 'y'
TEST_DIR = 'jpmml_test_data'

EPSILON = 0.00001

logging.basicConfig(format='%(asctime)s %(message)s')


# Adapted from http://stackoverflow.com/questions/1724693/find-a-file-in-python
def find_file_or_dir(name):
    for root, dirs, files in os.walk(os.path.dirname(__file__)):
        if name in files or name in dirs:
            return os.path.join(root, name)


class JPMMLTest():
    USE_VERIFICATION = False
    """
    If true, the PMML will be generated with the ModelVerification section that allows PMML interpreter to check the
    correctness of deserialized model.
    """

    def __init__(self):
        self.x = None
        self.y = None
        self.ctx = None
        self.converter = None

    @staticmethod
    def can_run():
        try:
            subprocess.check_call(['java', '-version'])
        except OSError:
            logging.warning("Couldn't find java to run JPMML integration tests")
            return False

        try:
            subprocess.check_call(['mvn', '-version'])
        except OSError:
            logging.warning("Couldn't find maven to run JPMML integration tests")
            return False

        return True

    @staticmethod
    def init_jpmml():
        result = subprocess.call(['mvn', '-q', 'clean', 'package', '-f', find_file_or_dir('jpmml-csv-evaluator')])
        assert result == 0, "Unable to package jpmml csv evaluator"
        return True

    # taken from http://stackoverflow.com/questions/18159221/remove-namespace-and-prefix-from-xml-in-python-using-lxml
    @staticmethod
    def remove_namespace(doc, namespace):
        ns = u'{%s}' % namespace
        nsl = len(ns)
        for elem in doc.getiterator():
            if elem.tag.startswith(ns):
                elem.tag = elem.tag[nsl:]

    @property
    def model(self):
        if self._model is None:
            raise NotImplementedError()
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def output(self):
        raise NotImplementedError()

    def setup_jpmml_test(self):
        if not JPMMLTest.can_run():
            logging.warning("Can't run regression test, java and/or maven not installed")
            return None

        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)
        os.makedirs(TEST_DIR)

        if self.USE_VERIFICATION:
            verification_data = self.x.copy()
            verification_data[TARGET_NAME] = self._model.predict_proba(self.x.values)[:, 1]

            xml = self.converter.pmml(verification_data=[
                dict((str(_[0]), _[1]) for _ in dict(row).items())
                for idx, row in verification_data[:10].iterrows()
            ]).toDOM().toprettyxml()
        else:
            xml = self.converter.pmml().toDOM().toprettyxml()

        pmml_hash = hashlib.md5(xml).hexdigest()
        pmml_file_path = os.path.join(TEST_DIR, pmml_hash + '.pmml')
        with open(pmml_file_path, 'w') as pmml_file:
            pmml_file.write(xml)

        input_file_path = os.path.join(TEST_DIR, pmml_hash + '_input.csv')
        self.x.to_csv(input_file_path, index=False)
        target_file_path = os.path.join(TEST_DIR, pmml_hash + '_output.csv')

        java_args = ' '.join(map("'{}'".format, [
            os.path.abspath(pmml_file_path),
            os.path.abspath(input_file_path),
            os.path.abspath(target_file_path)
        ]))
        result = subprocess.call([
            'mvn', 'package', 'exec:java', '-q', '-e',
            '-f', find_file_or_dir('jpmml-csv-evaluator'),
            '-Dexec.mainClass=sklearn.pmml.jpmml.JPMMLCSVEvaluator',
            '-Dexec.args=' + java_args
        ])
        if result:
            print(xml)
        assert result == 0, 'Executing JPMML evaluator returned non zero result'
        return pd.read_csv(target_file_path)

    def init_data(self):
        np.random.seed(12363)
        self.x = pd.DataFrame(np.random.randn(500, 4), columns=['col_' + str(_) for _ in range(4)])
        self.y = pd.DataFrame({TARGET_NAME: [np.random.choice([0, 1, 2]) for _ in range(self.x.shape[0])]})
        self._model.fit(self.x, np.ravel(self.y))
        self.ctx = TransformationContext()
        self.ctx.schemas[Schema.INPUT] = [RealNumericFeature(col) for col in list(self.x)]
        self.ctx.schemas[Schema.DERIVED] = []
        self.ctx.schemas[Schema.MODEL] = [RealNumericFeature(col) for col in list(self.x)]
        self.ctx.schemas[Schema.OUTPUT] = [self.output]

    def init_data_one_label(self):
        np.random.seed(12363)
        self.x = pd.DataFrame(np.random.randn(500, 4), columns=['col_' + str(_) for _ in range(4)])
        self.y = pd.DataFrame({TARGET_NAME: [np.random.choice([0, 1]) for _ in range(self.x.shape[0])]})
        self._model.fit(self.x, np.ravel(self.y))
        self.ctx = TransformationContext()
        self.ctx.schemas[Schema.INPUT] = [RealNumericFeature(col) for col in list(self.x)]
        self.ctx.schemas[Schema.DERIVED] = []
        self.ctx.schemas[Schema.MODEL] = [RealNumericFeature(col) for col in list(self.x)]
        self.ctx.schemas[Schema.OUTPUT] = [self.output]


class JPMMLRegressionTest(JPMMLTest):
    @property
    def output(self):
        return IntegerNumericFeature(name=TARGET_NAME)

    def test_regression(self):
        jpmml_predictions = self.setup_jpmml_test()
        if jpmml_predictions is None:
            return

        sklearn_predictions = pd.DataFrame({TARGET_NAME: self.converter.estimator.predict(self.x)})
        diff = jpmml_predictions[TARGET_NAME] - sklearn_predictions[TARGET_NAME]
        assert np.all(np.abs(diff) < EPSILON)


class JPMMLClassificationTest(JPMMLTest):
    @property
    def output(self):
        return StringCategoricalFeature(name=TARGET_NAME, value_list=["negative", "positive"])

    def test_classification(self):

        jpmml_predictions = self.setup_jpmml_test()
        if jpmml_predictions is None:
            return

        raw_sklearn_predictions = self.converter.estimator.predict_proba(self.x)
        prob_outputs = [self.output.name + '::' + str(clazz) for clazz in self.output.value_list]
        sklearn_predictions = pd.DataFrame(columns=prob_outputs)
        for index, prediction in enumerate(raw_sklearn_predictions):
            sklearn_predictions.loc[index] = list(prediction)

        np.testing.assert_almost_equal(
            np.array(jpmml_predictions[list(sklearn_predictions.columns)]),
            sklearn_predictions.values,
            err_msg='Probability mismatch'
        )
        np.testing.assert_equal(
            np.array(self.output.value_list)[self.converter.estimator.predict(self.x)],
            jpmml_predictions[self.output.name].values,
            err_msg='Labels mismatch'
        )