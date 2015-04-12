import functools
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.gradient_boosting import LogOddsEstimator
from sklearn.tree._tree import Tree, TREE_LEAF
from sklearn_pmml.convert import Converter, PMMLTransformationContext, Schema, find_converter
from sklearn_pmml.convert.features import *
from sklearn_pmml.convert.tree import DecisionTreeConverter
import sklearn_pmml.pmml as pmml
import numpy as np
from sklearn_pmml.convert import estimator_to_converter


class LogOddsEstimatorConverter(Converter):

    def is_applicable(self, obj, ctx):
        return True

    def transform(self, obj, ctx):
        super(LogOddsEstimatorConverter, self).transform(obj, ctx)
        assert isinstance(obj, LogOddsEstimator)
        rm = pmml.RegressionModel(functionName="regression", algorithmName="linearRegression")
        rm.append(self.mining_schema(ctx.schema))
        rm.append(pmml.RegressionTable(intercept=obj.prior))
        yield rm


class GradientBoostingConverter(Converter):

    def __init__(self, mode):
        super(GradientBoostingConverter, self).__init__()
        self.mode = mode

    def is_applicable(self, obj, ctx):
        return True

    def transform(self, obj, ctx):
        super(GradientBoostingConverter, self).transform(obj, ctx)
        assert isinstance(obj, GradientBoostingClassifier)
        dtc = DecisionTreeConverter(mode=DecisionTreeConverter.MODE_REGRESSION)
        inner_schema = Schema(ctx.schema.features, RealNumericFeature(name=ctx.schema.output.external_name))
        inner_ctx = PMMLTransformationContext(schema=inner_schema, metadata={})
        segmentation = pmml.Segmentation(multipleModelMethod="weightedAverage")

        init = pmml.Segment(weight=1)
        init.append(pmml.True_())
        for el in find_converter(obj.init_.__class__).transform(obj.init_, ctx):
            init.append(el)
        segmentation.append(init)

        for tm in map(functools.partial(dtc.transform, ctx=inner_ctx), obj.estimators_[:, 0]) :
            s = pmml.Segment(weight=obj.learning_rate)
            s.append(pmml.True_())
            for el in tm:
                s.append(el)
            segmentation.append(s)
        output = pmml.Output()
        output.append(pmml.OutputField(feature='predictedValue', name='predictedValue'))
        output_field = pmml.OutputField(
            dataType='double', feature='transformedValue',
            name=ctx.schema.output.external_name, optype=ctx.schema.output.optype
        )
        neg = pmml.Apply(function='*')
        neg.append(pmml.FieldRef(field='predictedValue'))
        neg.append(pmml.Constant(
            -(1 + obj.n_estimators * obj.learning_rate),
            dataType='double'
        ))
        exp = pmml.Apply(function='exp')
        exp.append(neg)
        plus = pmml.Apply(function='+')
        plus.append(pmml.Constant(1.0, dataType='double'))
        plus.append(exp)
        div = pmml.Apply(function='/')
        div.append(pmml.Constant(1.0, dataType='double'))
        div.append(plus)

        output_field.append(div)
        output.append(output_field)

        mm = pmml.MiningModel(functionName=self.MODE_REGRESSION)
        mm.append(self.mining_schema(ctx.schema))
        mm.append(output)
        mm.append(segmentation)
        yield mm


estimator_to_converter[GradientBoostingClassifier] = GradientBoostingConverter(mode=GradientBoostingConverter.MODE_CLASSIFICATION)
estimator_to_converter[LogOddsEstimator] = LogOddsEstimatorConverter()