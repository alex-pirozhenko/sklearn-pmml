import os
from unittest import TestCase
from sklearn.base import BaseEstimator

try:
    import cPickle as pickle
except:
    import pickle
from sklearn_pmml.convert import *
from sklearn_pmml import pmml


class TestSerializationMeta(type):
    TEST_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(TEST_DIR, 'data')
    ESTIMATOR_FILE_NAME = 'estimator.pkl'
    PMML_FILE_NAME = 'document.pmml'
    CONTEXT_FILE_NAME = 'context.pkl'

    def __new__(mcs, name, bases, d):
        """
        This method overrides default behaviour for creation of new instances. For every directory abc in data it
        creates a method called test_abc, with the body of load_and_compare function.
        """
        def gen_test(suite_name):
            def load_and_compare(self):
                # load the context.pkl, document.pmml and estimator.pkl
                suite_path = os.path.join(mcs.DATA_DIR, suite_name)
                content = os.listdir(suite_path)
                assert len(content) == 3, 'There should be exactly two files in the suite directory'
                assert mcs.ESTIMATOR_FILE_NAME in content, 'Estimator should be stored in {} file'.format(mcs.ESTIMATOR_FILE_NAME)
                assert mcs.PMML_FILE_NAME in content, 'PMML should be stored in {} file'.format(mcs.PMML_FILE_NAME)
                assert mcs.CONTEXT_FILE_NAME in content, 'Context should be stored in {} file'.format(mcs.CONTEXT_FILE_NAME)
                with open(os.path.join(suite_path, mcs.ESTIMATOR_FILE_NAME), 'r') as est_file:
                    est = pickle.load(est_file)
                    assert isinstance(est, BaseEstimator), '{} should be a trained estimator'.format(mcs.ESTIMATOR_FILE_NAME)
                with open(os.path.join(suite_path, mcs.CONTEXT_FILE_NAME), 'r') as ctx_file:
                    ctx = pickle.load(ctx_file)
                    assert isinstance(ctx, TransformationContext), '{} should be a transformation context'.format(mcs.CONTEXT_FILE_NAME)
                converter = find_converter(est)
                assert converter is not None, 'Can not find converter for {}'.format(est)
                transformed_pmml = converter(est, ctx).pmml()
                with open(os.path.join(suite_path, mcs.PMML_FILE_NAME), 'r') as pmml_file:
                    loaded_pmml = pmml.CreateFromDocument('\n'.join(pmml_file.readlines()))
                self.maxDiff = None
                # make sure that the expected PMML matches the produced one
                self.assertEquals(loaded_pmml.toDOM().toprettyxml(), transformed_pmml.toDOM().toprettyxml())

            return load_and_compare

        # for every batch in the data dir create a corresponding test method
        for case in os.listdir(TestSerializationMeta.DATA_DIR):
            test_name = 'test_{}'.format(case)
            d[test_name] = gen_test(case)
        return type.__new__(mcs, name, bases, d)


class TestSerialization(TestCase):
    """
    This is an automated tester for serializers. It uses a custom metaclass to define the test cases based on the
    content of the data directory. For the logic behind every check see load_and_compare method above.
    """
    __metaclass__ = TestSerializationMeta


