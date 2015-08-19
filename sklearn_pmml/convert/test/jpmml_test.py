__author__ = 'evancox'

import numpy as np
import hashlib
import os
import shutil
import subprocess
import logging
from lxml import etree

from sklearn_pmml.convert import TransformationContext
from sklearn_pmml.convert.features import *


_TARGET = [0, 1, 2]
_TARGET_NAME = 'y'
_TEST_DIR = 'jpmml_test_data'

logging.basicConfig(format='%(asctime)s %(message)s')

#Adapted from http://stackoverflow.com/questions/1724693/find-a-file-in-python
def find_file_or_dir(name, base_path=os.path.dirname(__file__)):
    for root, dirs, files in os.walk(base_path):
        if name in files or name in dirs:
            return os.path.join(root, name)


class JPMMLTest():

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
            logging.warning("Couldn't find java to run JPMML integration tests")
            return False

        return True



    @staticmethod
    def init_jpmml():
        result = subprocess.call(['mvn', '-q', 'clean', 'package', '-f', find_file_or_dir('jpmml-csv-evaluator')])
        assert result == 0, "Unable to package jpmml csv evaluator"
        return True

    #taken from http://stackoverflow.com/questions/18159221/remove-namespace-and-prefix-from-xml-in-python-using-lxml
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

        if os.path.exists(_TEST_DIR):
            shutil.rmtree(_TEST_DIR)
        os.makedirs(_TEST_DIR)

        #evancox This is a hack to allow us to mark the version down as 4.1 so we can run with the BSD version of JPMML rather than the AGPL one
        xml = self.converter.pmml().toxml("utf-8")
        pmml = etree.fromstring(xml)
        pmml.set('version', '4.1')
        JPMMLTest.remove_namespace(pmml, 'http://www.dmg.org/PMML-4_2')
        xml = etree.tostring(pmml, pretty_print=True)
        xml = xml.replace('http://www.dmg.org/PMML-4_2', 'http://www.dmg.org/PMML-4_1')

        pmml_hash = hashlib.md5(xml).hexdigest()
        pmml_file_path = os.path.join(_TEST_DIR, pmml_hash + '.pmml')
        with open(pmml_file_path, 'w') as pmml_file:
            pmml_file.write(xml)

        input_file_path = os.path.join(_TEST_DIR, pmml_hash + '_input.csv')
        self.x.to_csv(input_file_path,index=False)
        target_file_path = os.path.join(_TEST_DIR, pmml_hash + '_output.csv')

        java_args = ' '.join(
            "'" + _ + "'"
            for _ in [
                os.path.abspath(pmml_file_path),
                os.path.abspath(input_file_path),
                os.path.abspath(target_file_path)
            ]
        )
        result = subprocess.call([
            'mvn', 'exec:java', '-q', '-f', find_file_or_dir('jpmml-csv-evaluator'),
            '-Dexec.mainClass=sklearn.pmml.jpmml.JPMMLCSVEvaluator',
            '-Dexec.args=' + java_args
        ])
        self.assertEqual(result, 0, 'Executing jpmml evaluator returned non zero result')
        return pd.read_csv(target_file_path)

    def init_data(self):
        np.random.seed(12363)
        self.x = pd.DataFrame(np.random.randn(500, 10))
        self.y = pd.DataFrame({_TARGET_NAME:[np.random.choice([0, 1, 2]) for _ in range(self.x.shape[0])]})
        self._model.fit(self.x, np.ravel(self.y))
        self.ctx = TransformationContext(
            input=[RealNumericFeature(col) for col in list(self.x)],
            derived=[],
            model=[RealNumericFeature(col) for col in list(self.x)],
            output=[self.output]
        )





class JPMMLRegressionTest(JPMMLTest):

    @property
    def output(self):
        return IntegerNumericFeature(_TARGET_NAME, _TARGET)

    def test_regression(self):
        jpmml_predictions = self.setup_jpmml_test()
        if jpmml_predictions is None:
            return

        sklearn_predictions = pd.DataFrame({_TARGET_NAME:self.converter.estimator.predict(self.x)})
        diff = jpmml_predictions[_TARGET_NAME] - sklearn_predictions[_TARGET_NAME]
        self.assertTrue(np.all(np.abs(diff) < .001))


class JPMMLClassificationTest(JPMMLTest):

    @property
    def output(self):
        return IntegerCategoricalFeature(_TARGET_NAME, _TARGET)

    def test_classification(self):

        jpmml_predictions = self.setup_jpmml_test()
        if jpmml_predictions is None:
            return

        raw_sklearn_predictions = self.converter.estimator.predict_proba(self.x)
        prob_outputs = ['Probability_' + str(clazz) for clazz in self.converter.estimator.classes_]
        sklearn_predictions = pd.DataFrame(columns=prob_outputs)
        for index, prediction in enumerate(raw_sklearn_predictions):
            sklearn_predictions.loc[index] = list(prediction)
        self.assertTrue(np.all(jpmml_predictions[prob_outputs] == sklearn_predictions))



