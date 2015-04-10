from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn_pmml import PMMLTransformationContext
from sklearn_pmml.convert import Schema, PMMLBuilder
from sklearn_pmml.convert.features import IntegerNumericFeature, IntegerCategoricalFeature, StringCategoricalFeature
from sklearn_pmml.convert.tree import DecisionTreeConverter


est = DecisionTreeRegressor(max_depth=2)
est.fit([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], [0, 1, 1, 1])
ctx = PMMLTransformationContext(
    schema=Schema(features=[IntegerNumericFeature('x1'), StringCategoricalFeature('x2', ['zero', 'one']), ],
                  output=IntegerNumericFeature('output')), metadata={})

print PMMLBuilder().build(est, ctx).toDOM().toprettyxml()

