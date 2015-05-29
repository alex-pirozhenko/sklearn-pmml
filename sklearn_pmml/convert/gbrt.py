from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble.gradient_boosting import LogOddsEstimator

from sklearn_pmml.convert.model import EstimatorConverter
from sklearn_pmml.convert.tree import DecisionTreeConverter
import sklearn_pmml.pmml as pmml
from sklearn_pmml.convert.utils import estimator_to_converter, find_converter


class LogOddsEstimatorConverter(EstimatorConverter):
    REGRESSION_LINEAR = "linearRegression"

    def __init__(self, estimator, context):
        super(LogOddsEstimatorConverter, self).__init__(estimator, context, self.MODE_REGRESSION)

        assert isinstance(estimator, LogOddsEstimator), 'This converter can only process LogOddsEstimator instances'

    def model(self, verification_data=None):
        rm = pmml.RegressionModel(functionName=self.model_function_name, algorithmName=self.REGRESSION_LINEAR)
        rm.append(self.mining_schema())
        rm.append(pmml.RegressionTable(intercept=self.estimator.prior))
        if verification_data is not None:
            rm.append(self.model_verification(verification_data))
        return rm


class GradientBoostingConverter(EstimatorConverter):
    def __init__(self, estimator, context):
        super(GradientBoostingConverter, self).__init__(estimator, context, self.MODE_CLASSIFICATION)

        assert isinstance(estimator, GradientBoostingClassifier), \
            'This converter can only process GradientBoostingClassifier instances'
        assert len(context.schemas[self.SCHEMA_OUTPUT]) == 1, 'Only one-label classification is supported'
        assert not estimator.loss_.is_multi_class, 'Only one-label classification is supported'
        assert context.schemas[self.SCHEMA_OUTPUT][0].data_type == 'double', 'PMML version only returns probabilities'
        assert context.schemas[self.SCHEMA_OUTPUT][0].optype == 'continuous', 'PMML version only returns probabilities'
        assert find_converter(estimator.init_) is not None, 'Can not find a converter for {}'.format(estimator.init_)

    def model(self, verification_data=None):
        # gradient boosting is always a regression model in PMML terms:
        mining_model = pmml.MiningModel(functionName=self.MODE_REGRESSION)
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
        output.append(pmml.OutputField(feature='predictedValue', name='predictedValue'))
        output_feature = self.context.schemas[self.SCHEMA_OUTPUT][0]
        output_field = pmml.OutputField(
            dataType='double', feature='transformedValue',
            name=output_feature.full_name, optype=output_feature.optype
        )
        neg = pmml.Apply(function='*')
        neg.append(pmml.FieldRef(field='predictedValue'))
        neg.append(pmml.Constant(
            # there is no notion of weighted sum in segment aggregation, so we used weighted average,
            # and now the result should be multiplied by total weight
            -(1 + self.estimator.n_estimators * self.estimator.learning_rate),
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
        return output

    def segmentation(self):
        """
        Build a segmentation (sequence of estimators)
        :return: Segmentation element
        """
        # there is no notion of weighted sum, so we should take weighted average and multiply result by total weight
        # in output transformation
        segmentation = pmml.Segmentation(multipleModelMethod="weightedAverage")

        # first, transform initial estimator
        init_segment = pmml.Segment(weight=1)
        init_segment.append(pmml.True_())
        init_segment.append(find_converter(self.estimator.init_)(self.estimator.init_, self.context).model())
        segmentation.append(init_segment)

        for est in self.estimator.estimators_[:, 0]:
            s = pmml.Segment(weight=self.estimator.learning_rate)
            s.append(pmml.True_())
            s.append(DecisionTreeConverter(est, self.context, self.MODE_REGRESSION)._model())
            segmentation.append(s)

        return segmentation


estimator_to_converter[GradientBoostingClassifier] = GradientBoostingConverter
estimator_to_converter[LogOddsEstimator] = LogOddsEstimatorConverter