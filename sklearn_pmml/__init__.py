from sklearn_pmml.convert import PMMLTransformationContext
from sklearn.base import BaseEstimator


__all__ = ['to_pmml']


def to_pmml(estimator, schema, **metadata):
    assert isinstance(estimator, BaseEstimator)
    ctx = PMMLTransformationContext(schema, metadata)
    raise NotImplementedError()