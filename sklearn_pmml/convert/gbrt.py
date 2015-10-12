from copy import copy
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble.gradient_boosting import LogOddsEstimator

from sklearn_pmml.convert.features import *
from sklearn_pmml.convert.model import EstimatorConverter, ClassifierConverter, ModelMode, RegressionConverter, Schema, \
    TransformationContext
from sklearn_pmml.convert.tree import DecisionTreeConverter
import sklearn_pmml.pmml as pmml
from sklearn_pmml.convert.utils import estimator_to_converter, find_converter


class LogOddsEstimatorConverter(RegressionConverter):
    REGRESSION_LINEAR = "linearRegression"

    def __init__(self, estimator, context):
        super(LogOddsEstimatorConverter, self).__init__(estimator, context)

        assert isinstance(estimator, LogOddsEstimator), 'This converter can only process LogOddsEstimator instances'

    def model(self, verification_data=None):
        rm = pmml.RegressionModel(functionName=self.model_function.value, algorithmName=self.REGRESSION_LINEAR)
        rm.append(self.mining_schema())
        rm.append(pmml.RegressionTable(intercept=self.estimator.prior))
        if verification_data is not None:
            rm.append(self.model_verification(verification_data))
        return rm


class GradientBoostingConverter(ClassifierConverter):
    """
    Converter for GradientBoostingClassifier model.

    NOTE: at the moment only binary one-label classification is supported.
    """
    SCHEMAS_IN_MINING_MODEL = {Schema.INPUT}

    def __init__(self, estimator, context):
        super(GradientBoostingConverter, self).__init__(estimator, context)

        assert isinstance(estimator, GradientBoostingClassifier), \
            'This converter can only process GradientBoostingClassifier instances'
        assert len(context.schemas[Schema.OUTPUT]) == 1, 'Only one-label classification is supported'
        assert not estimator.loss_.is_multi_class, 'Only one-label classification is supported'
        assert context.schemas[Schema.OUTPUT][0].optype == FeatureOpType.CATEGORICAL, \
            'Classification output must be categorical'
        assert len(context.schemas[Schema.OUTPUT][0].value_list) == 2, 'Only binary classifier is supported'
        assert find_converter(estimator.init_) is not None, 'Can not find a converter for {}'.format(estimator.init_)

    def model(self, verification_data=None):
        # The ensemble of regression models can only be a regression model. Surprise!
        mining_model = pmml.MiningModel(functionName=ModelMode.REGRESSION.value)
        mining_model.append(self.mining_schema())
        mining_model.append(self.output_transformation())
        mining_model.append(self.segmentation())
        if verification_data is not None:
            mining_model.append(self.model_verification(verification_data))
        return mining_model

    def output_transformation(self):
        """
        Build sigmoid output transformation:
        proba = 1 / (1 + exp(-(initial_estimate + weighted_sum(estimates))))
        :return: Output element
        """
        output = pmml.Output()

        # storing the raw prediction into internal::varname variable
        for f in self.context.schemas[Schema.INTERNAL]:
            output.append(pmml.OutputField(feature='predictedValue', name=Schema.INTERNAL.extract_feature_name(f)))

        # setting up a logistic transformation for the positive label
        positive_category = self.context.schemas[Schema.CATEGORIES][1]
        output_field = pmml.OutputField(
            dataType=positive_category.data_type.value,
            feature='transformedValue',
            name=Schema.CATEGORIES.extract_feature_name(positive_category),
            optype=positive_category.optype.value
        )
        neg = pmml.Apply(function='*')
        neg.append(pmml.FieldRef(field=Schema.INTERNAL.extract_feature_name(positive_category.namespace)))
        neg.append(pmml.Constant(
            # there is no notion of weighted sum in segment aggregation, so we used weighted average,
            # and now the result should be multiplied by total weight
            -(1 + self.estimator.n_estimators * self.estimator.learning_rate),
            dataType=FeatureType.DOUBLE.value
        ))
        exp = pmml.Apply(function='exp')
        exp.append(neg)
        plus = pmml.Apply(function='+')
        plus.append(pmml.Constant(1.0, dataType=FeatureType.DOUBLE.value))
        plus.append(exp)
        div = pmml.Apply(function='/')
        div.append(pmml.Constant(1.0, dataType=FeatureType.DOUBLE.value))
        div.append(plus)
        output_field.append(div)
        output.append(output_field)

        # probability of negative label is 1 - positive_proba
        negative_category = self.context.schemas[Schema.CATEGORIES][0]
        output_field = pmml.OutputField(
            dataType=negative_category.data_type.value,
            feature='transformedValue',
            name=Schema.CATEGORIES.extract_feature_name(negative_category),
            optype=negative_category.optype.value
        )
        subtract = pmml.Apply(function='-')
        subtract.append(pmml.Constant(1, dataType=FeatureType.DOUBLE.value))
        subtract.append(pmml.FieldRef(field=Schema.CATEGORIES.extract_feature_name(positive_category)))
        output_field.append(subtract)
        output.append(output_field)

        # now we should define a label; we can look at the raw predicted output and compare it with 0
        label = self.context.schemas[Schema.OUTPUT][0]
        output_field = pmml.OutputField(
            feature='transformedValue',
            name=Schema.OUTPUT.extract_feature_name(label),
            optype=label.optype.value,
            dataType=label.data_type.value
        )
        discretize = pmml.Discretize(field=Schema.INTERNAL.extract_feature_name(label))
        discretize_bin = pmml.DiscretizeBin(binValue=label.value_list[0])
        discretize_bin.append(pmml.Interval(closure="openOpen", rightMargin=0))
        discretize.append(discretize_bin)
        discretize_bin = pmml.DiscretizeBin(binValue=label.value_list[1])
        discretize_bin.append(pmml.Interval(closure="closedOpen", leftMargin=0))
        discretize.append(discretize_bin)
        output_field.append(discretize)
        output.append(output_field)

        return output

    def segmentation(self):
        """
        Build a segmentation (sequence of estimators)
        :return: Segmentation element
        """
        # there is no notion of weighted sum, so we should take weighted average and multiply result by total weight
        # in output transformation
        segmentation = pmml.Segmentation(multipleModelMethod="weightedAverage")

        # build the context for the nested regression models by replacing output categorical feature
        # with the continuous numeric feature
        regression_context = TransformationContext(schemas=dict(self.context.schemas))
        regression_context.schemas[Schema.OUTPUT] = [RealNumericFeature(
            name=self.context.schemas[Schema.OUTPUT][0].name,
            namespace=Schema.NUMERIC.namespace
        )]

        # first, transform initial estimator
        init_segment = pmml.Segment(weight=1)
        init_segment.append(pmml.True_())
        init_segment.append(find_converter(self.estimator.init_)(self.estimator.init_, regression_context).model())
        segmentation.append(init_segment)

        for est in self.estimator.estimators_[:, 0]:
            s = pmml.Segment(weight=self.estimator.learning_rate)
            s.append(pmml.True_())
            s.append(DecisionTreeConverter(est, regression_context, ModelMode.REGRESSION)._model())
            segmentation.append(s)

        return segmentation


estimator_to_converter[GradientBoostingClassifier] = GradientBoostingConverter
estimator_to_converter[LogOddsEstimator] = LogOddsEstimatorConverter