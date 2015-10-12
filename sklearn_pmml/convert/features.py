from enum import Enum
import pandas as pd


class FeatureOpType(Enum):
    CATEGORICAL = 'categorical'
    CONTINUOUS = 'continuous'


class FeatureType(Enum):
    DOUBLE = 'double'
    INT = 'integer'
    STRING = 'string'


class InvalidValueTreatment(Enum):
    AS_IS = 'asIs'


class Feature(object):
    def __init__(self, name, namespace='', invalid_value_treatment=InvalidValueTreatment.AS_IS):
        """
        Create a new feature
        :type name: str
        :type namespace: str
        :type invalid_value_treatment: InvalidValueTreatment
        """
        self._name = str(name)
        self._namespace = str(namespace)
        self._invalid_value_treatment = invalid_value_treatment

    @property
    def name(self):
        """
        :rtype: str
        """
        return self._name

    @property
    def namespace(self):
        """
        :rtype: str
        """
        return self._namespace

    @property
    def full_name(self):
        """
        :rtype: str
        """
        if self._namespace:
            return '{}.{}'.format(self._namespace, self.name)
        else:
            return self.name

    @property
    def invalid_value_treatment(self):
        return self._invalid_value_treatment

    @property
    def optype(self):
        """
        :rtype: FeatureOpType
        """
        raise NotImplementedError()

    @property
    def data_type(self):
        """
        :rtype: FeatureType
        """
        raise NotImplementedError()

    def from_number(self, value):
        raise NotImplementedError()

    def __str__(self):
        return self.name

    def __repr__(self):
        return "{}#{}".format(self.name, self.__class__.__name__)


class NumericFeature(Feature):
    @property
    def optype(self):
        return FeatureOpType.CONTINUOUS

    def from_number(self, value):
        return float(value)


class RealNumericFeature(NumericFeature):
    @property
    def data_type(self):
        return FeatureType.DOUBLE


class IntegerNumericFeature(NumericFeature):
    def from_number(self, value):
        return int(value)

    @property
    def data_type(self):
        return FeatureType.INT


class CategoricalFeature(Feature):
    """
    Represents a categorical feature. Categorical features are defined with optype 'categorical' and the corresponding
    dataType. The corresponding derived field will have a double data type and will be defined as a MapValues PMML
    element.
    """
    def __init__(self, name, value_list, namespace='', invalid_value_treatment=InvalidValueTreatment.AS_IS):
        super(CategoricalFeature, self).__init__(name, namespace, invalid_value_treatment)
        self.value_list = value_list

    @property
    def optype(self):
        return FeatureOpType.CATEGORICAL

    def from_number(self, value):
        assert value >= 0, 'Negative numbers can not be used as categorical indexes'
        assert value < len(self.value_list), 'Unknown category index {}'.format(value)
        return self.value_list[value]

    def to_number(self, value):
        """
        Transform categorical value to the ordinal. Raises ValueError if value is not in self.value_list
        """
        return list(self.value_list).index(value)


class IntegerCategoricalFeature(CategoricalFeature):
    @property
    def data_type(self):
        return FeatureType.INT


class StringCategoricalFeature(CategoricalFeature):
    @property
    def data_type(self):
        return FeatureType.STRING


class DerivedFeature(NumericFeature):
    """
    This class represents a derived feature constructed from previously defined features.
    The transformation parameter defines the recipe for creating a feature, and will be inserted into pmml.DerivedField
    element for this feature.
    Note, that the transformation only allows references to the already declare fields.

    For convenience, one can also pass the function that performs the transformation on the input data frame.
    """

    def __init__(self, feature, transformation, function):
        """
        Construct a derived feature.
        :param feature: declaration of feature (name, data_type and optype)
        :type feature: Feature
        :param transformation: definition of DerivedField content
        :param function: transformation function
        :type function: callable
        """
        super(DerivedFeature, self).__init__(
            name=feature.name,
            namespace=feature.namespace,
            invalid_value_treatment=feature.invalid_value_treatment
        )
        assert isinstance(feature, NumericFeature), 'All derived features must be declared as NumericFeatures'
        assert function is not None, 'Function can not be None'
        assert callable(function), 'Function must be callable'
        self.feature = feature
        self.transformation = transformation
        self.function = function

    def from_number(self, value):
        return self.feature.from_number(value)

    @property
    def data_type(self):
        return self.feature.data_type

    @property
    def optype(self):
        return self.feature.optype

    def apply(self, df):
        """
        Calculate derived feature's values based on the values in the input data frame.
        Note that the input data frame will not be affected by the transformation.
        :param df: input data frame
        :return: array with results
        """
        assert self.function is not None, 'Function was not provided'
        assert isinstance(df, pd.DataFrame), 'Input should be a data frame'
        return self.function(df.copy(deep=False))