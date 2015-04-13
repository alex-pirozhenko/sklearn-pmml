class Feature(object):
    INVALID_TREATMENT_AS_IS = 'asIs'

    def __init__(self, name, namespace='', invalid_value_treatment=INVALID_TREATMENT_AS_IS):
        self._name = name
        self._namespace = namespace
        self._invalid_value_treatment = invalid_value_treatment

    @property
    def name(self):
        return self._name

    @property
    def full_name(self):
        if self._namespace:
            return '{}::{}'.format(self._namespace, self.name)
        else:
            return self.name

    @property
    def invalid_value_treatment(self):
        return self._invalid_value_treatment

    @property
    def optype(self):
        raise NotImplementedError()

    @property
    def data_type(self):
        raise NotImplementedError()

    def from_number(self, value):
        raise NotImplementedError()


class NumericFeature(Feature):
    @property
    def optype(self):
        return "continuous"

    def from_number(self, value):
        return float(value)


class RealNumericFeature(NumericFeature):
    @property
    def data_type(self):
        return "double"


class IntegerNumericFeature(NumericFeature):
    def from_number(self, value):
        return int(value)

    @property
    def data_type(self):
        return "integer"


class CategoricalFeature(Feature):
    def __init__(self, name, value_list):
        super(CategoricalFeature, self).__init__(name)
        self.value_list = value_list

    @property
    def optype(self):
        return "categorical"

    def from_number(self, value):
        assert value >= 0, 'Negative numbers can not be used as categorical indexes'
        assert value < len(self.value_list), 'Unknown category index {}'.format(value)
        return self.value_list[value]


class IntegerCategoricalFeature(CategoricalFeature):
    @property
    def data_type(self):
        return "integer"


class StringCategoricalFeature(CategoricalFeature):
    @property
    def data_type(self):
        return "string"