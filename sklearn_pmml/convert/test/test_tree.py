from sklearn.tree import DecisionTreeClassifier
from sklearn_pmml import PMMLTransformationContext
from sklearn_pmml.convert import Schema, PMMLBuilder
from sklearn_pmml.convert.features import IntegerNumericFeature, IntegerCategoricalFeature, StringCategoricalFeature
from sklearn_pmml.convert.tree import DecisionTreeClassifierConverter


est = DecisionTreeClassifier(max_depth=2)
est.fit([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], [0, 1, 1, 1])
ctx = PMMLTransformationContext(
    schema=Schema(features=[IntegerNumericFeature('x1'), StringCategoricalFeature('x2', ['zero', 'one']), ],
                  output=StringCategoricalFeature('output', ['neg', 'pos'])), metadata={})
print DecisionTreeClassifierConverter(
    mode=DecisionTreeClassifierConverter.MODE_CLASSIFICATION
).transform(est, ctx)

print PMMLBuilder().build(est, ctx).toDOM().toprettyxml()