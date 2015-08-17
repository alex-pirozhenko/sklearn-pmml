__author__ = 'evancox'


from sklearn.ensemble import RandomForestClassifier
from sklearn_pmml.convert.model import EstimatorConverter
from sklearn_pmml.convert.tree import DecisionTreeConverter
from sklearn_pmml.convert.utils import estimator_to_converter

import sklearn_pmml.pmml as pmml

class RandomForestClassifierConverter(EstimatorConverter):
    def __init__(self, estimator, context):
        super(RandomForestClassifierConverter, self).__init__(estimator, context, self.MODE_CLASSIFICATION)
        assert isinstance(estimator, RandomForestClassifier), \
            'This converter can only process RandomForestClassifier instances'
        assert len(context.schemas[self.SCHEMA_OUTPUT]) == 1, 'Only one-label classification is supported'

    def model(self, verification_data=None):
        mining_model = pmml.MiningModel(functionName=self.MODE_CLASSIFICATION)
        mining_model.append(self.mining_schema())
        mining_model.append(self.output())
        mining_model.append(self.segmentation())
        if verification_data is not None:
            mining_model.append(self.model_verification(verification_data))
        return mining_model


    def segmentation(self):
        """
        Build a segmentation (sequence of estimators)
        :return: Segmentation element
        """
        segmentation = pmml.Segmentation(multipleModelMethod="majorityVote")

        for index, est in enumerate(self.estimator.estimators_):
            s = pmml.Segment(id=index)
            s.append(pmml.True_())
            s.append(DecisionTreeConverter(est, self.context, self.MODE_CLASSIFICATION)._model())
            segmentation.append(s)

        return segmentation


estimator_to_converter[RandomForestClassifier] = RandomForestClassifierConverter