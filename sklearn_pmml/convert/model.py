from collections import defaultdict
from enum import Enum
from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator
from sklearn_pmml import pmml
from sklearn_pmml.convert.utils import pmml_row, assert_equal
from sklearn_pmml.convert.features import *
from pyxb.utils.domutils import BindingDOMSupport as bds
import numpy as np


class TransformationContext(object):
    """
    Context holder object
    """

    def __init__(self, schemas=None):
        """
        :type schemas: dict[Schema, list[Feature]] | None
        """
        if schemas is None:
            schemas = {}
        self.schemas = schemas


class ModelMode(Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'


class Schema(Enum):
    INPUT = ('input', True, True)
    """
    Schema used to define input variables. Short names allowed
    """

    OUTPUT = ('output', True, True)
    """
    Schema used to define output variables. Short names allowed. For the categorical variables the continuous
    probability variables will be automatically created as <feature_name>.<feature_value>
    """

    DERIVED = ('derived', False, False)
    """
    Schema used to define derived features. Short names not allowed due to potential overlap with input variables.
    """

    NUMERIC = ('numeric', False, False)
    """
    Schema used to encode categorical features as numbers. Short names not allowed due to their overlap with
    input variables
    """

    MODEL = ('model', True, False)
    """
    Schema used to define features fed into the sklearn estimator.
    Short names allowed because these variables are not going into PMML.
    """

    INTERNAL = ('internal', False, True)
    """
    This schema may be used by complex converters to hide the variables used for internal needs
    (e.g. the raw predictions of GBRT)
    """

    CATEGORIES = ('categories', True, False)
    """
    This schema is used to extend categorical outputs with probabilities of categories
    """

    def __init__(self, name, short_names_allowed, data_dict_elibigle):
        self._name = name
        self._short_names_allowed = short_names_allowed
        self._data_dict_elibigle = data_dict_elibigle

    @property
    def namespace(self):
        """
        The namespace corresponding to the schema
        """
        return self._name

    @property
    def short_names_allowed(self):
        """
        The schema allows usage of short names instead of fully-qualified names
        """
        return self._short_names_allowed

    @property
    def eligible_for_data_dictionary(self):
        """
        The variables defined in the schema should appear in the DataDictionary
        """
        return self._data_dict_elibigle

    def extract_feature_name(self, f):
        """
        Extract the printed name of the feature.
        :param f: feature to work with
        :type f: Feature|str
        """
        if self.short_names_allowed:
            if isinstance(f, str):
                return f
            else:
                return f.full_name
        else:
            return "{}.{}".format(self.namespace, f if isinstance(f, str) else f.full_name)


class EstimatorConverter(object):
    """
    A new base class for the estimator converters
    """
    EPSILON = 0.00001
    SCHEMAS_IN_MINING_MODEL = {Schema.INPUT, Schema.INTERNAL}

    def __init__(self, estimator, context, mode):
        self.model_function = mode
        self.estimator = estimator
        self.context = context

        assert not any(isinstance(_, DerivedFeature) for _ in context.schemas[Schema.INPUT]), \
            'Input schema represents the input fields only'
        assert all(isinstance(_, DerivedFeature) for _ in context.schemas[Schema.DERIVED]), \
            'Derived schema represents the set of automatically generated fields'
        assert not any(isinstance(_, DerivedFeature) for _ in context.schemas[Schema.OUTPUT]), \
            'Only regular features allowed in output schema; use Output transformation if you want to transform values'

        # create a new schema for categories probabilities
        categories = []
        for feature in context.schemas[Schema.OUTPUT]:
            if isinstance(feature, CategoricalFeature):
                for value in feature.value_list:
                    categories.append(RealNumericFeature(
                        name=value,
                        namespace=feature.name
                    ))
        context.schemas[Schema.CATEGORIES] = categories

    def data_dictionary(self):
        """
        Build a data dictionary and return a DataDictionary element.

        DataDictionary contains feature types for all variables used in the PMML,
        except the ones defined as Derived Features
        """
        dd = pmml.DataDictionary()
        for schema, fields in sorted(self.context.schemas.items(), key=lambda x: x[0].name):
            assert isinstance(schema, Schema)
            if schema.eligible_for_data_dictionary:
                for f in fields:
                    data_field = pmml.DataField(
                        dataType=f.data_type.value,
                        name=schema.extract_feature_name(f),
                        optype=f.optype.value)
                    dd.DataField.append(data_field)
                    if isinstance(f, CategoricalFeature):
                        for v in f.value_list:
                            data_field.append(pmml.Value(value_=v))
        return dd

    def output(self):
        """
        Output section of PMML contains all model outputs.
        :return: pmml.Output
        """
        output = pmml.Output()

        # the response variables
        for feature in self.context.schemas[Schema.OUTPUT]:
            output_field = pmml.OutputField(
                name=Schema.OUTPUT.extract_feature_name(feature),
                feature='predictedValue'
            )
            output.append(output_field)

        return output

    def transformation_dictionary(self):
        """
        Build a transformation dictionary and return a TransformationDictionary element
        """
        td = pmml.TransformationDictionary()
        # define a schema with all variables available for a model
        encoded_schema = []
        self.context.schemas[Schema.NUMERIC] = encoded_schema
        idx = {}

        # First, populate transformation dictionary for _all_ derived fields, because they can be requested later
        for f in self.context.schemas[Schema.DERIVED]:
            ef = RealNumericFeature(name=f.name)
            df = pmml.DerivedField(
                name=ef.full_name,
                optype=ef.optype.value,
                dataType=ef.data_type.value
            )
            df.append(f.transformation)
            td.append(df)
            assert f.name not in idx, 'Duplicate field definition: {}'.format(f.name)
            idx[f.name] = ef

        # second, define the numeric transformations for the categorical variables
        for f in self.context.schemas[Schema.INPUT]:
            assert f.name not in idx, 'Duplicate field definition: {}'.format(f.name)
            if isinstance(f, CategoricalFeature):
                ef = RealNumericFeature(name=f.name, namespace=Schema.NUMERIC.namespace)
                # create a record in transformation dictionary with mapping from raw values into numbers
                df = pmml.DerivedField(
                    name=ef.full_name,
                    optype=ef.optype.value,
                    dataType=ef.data_type.value
                )
                mv = pmml.MapValues(outputColumn='output', dataType=ef.data_type.value)
                mv.append(pmml.FieldColumnPair(field=f.full_name, column='input'))
                it = pmml.InlineTable()
                for i, v in enumerate(f.value_list):
                    it.append(pmml_row(input=v, output=i))
                td.append(df.append(mv.append(it)))
                idx[f.name] = ef
            else:
                idx[f.name] = f

        # now we can build a mirror of model schema into the numeric schema
        self.context.schemas[Schema.NUMERIC] = [idx[f.name] for f in self.context.schemas[Schema.MODEL]]

        return td

    def model(self, verification_data=None):
        """
        Build a mining model and return one of the MODEL-ELEMENTs
        """
        pass

    def model_verification(self, verification_data):
        """
        Use the input verification_data, apply the transformations, evaluate the model response and produce the
        ModelVerification element
        :param verification_data: list of dictionaries or data frame
        :type verification_data: dict[str, object]|pd.DataFrame
        :return: ModelVerification element
        """
        verification_data = pd.DataFrame(verification_data)
        assert len(verification_data) > 0, 'Verification data can not be empty'

        verification_input = pd.DataFrame(index=verification_data.index)
        verification_model_input = pd.DataFrame(index=verification_data.index)
        for key in self.context.schemas[Schema.INPUT]:
            # all input features MUST be present in the verification_data
            assert key.full_name in verification_data.columns, 'Missing input field "{}"'.format(key.full_name)
            verification_input[Schema.INPUT.extract_feature_name(key)] = verification_data[key.full_name]
            if isinstance(key, CategoricalFeature):
                verification_model_input[Schema.INPUT.extract_feature_name(key)] = np.vectorize(key.to_number)(verification_data[key.full_name])
            else:
                verification_model_input[Schema.INPUT.extract_feature_name(key)] = verification_data[key.full_name]

        for key in self.context.schemas[Schema.DERIVED]:
            assert isinstance(key, DerivedFeature), 'Only DerivedFeatures are allowed in the DERIVED schema'
            verification_model_input[key.full_name] = key.apply(verification_input)

        # at this point we can check that MODEL schema contains only known features
        for key in self.context.schemas[Schema.MODEL]:
            assert Schema.MODEL.extract_feature_name(key) in verification_model_input.columns, \
                'Unknown feature "{}" in the MODEL schema'.format(key.full_name)

        # TODO: we can actually support multiple columns, but need to figure out the way to extract the data
        # TODO: from the estimator properly
        # building model results
        assert len(self.context.schemas[Schema.OUTPUT]) == 1, 'Only one output is currently supported'
        key = self.context.schemas[Schema.OUTPUT][0]
        model_input = verification_model_input[list(map(Schema.MODEL.extract_feature_name, self.context.schemas[Schema.MODEL]))].values
        model_results = np.vectorize(key.from_number)(self.estimator.predict(X=model_input))
        if key.full_name in verification_data:
            # make sure that if results are provided, the expected and actual values are equal
            assert_equal(key, model_results, verification_data[key.full_name].values)
        verification_input[Schema.OUTPUT.extract_feature_name(key)] = model_results

        if isinstance(key, CategoricalFeature):
            probabilities = self.estimator.predict_proba(X=model_input)
            for i, key in enumerate(self.context.schemas[Schema.CATEGORIES]):
                verification_input[Schema.CATEGORIES.extract_feature_name(key)] = probabilities[:, i]

        fields = []
        field_names = []
        for s in [Schema.INPUT, Schema.OUTPUT, Schema.CATEGORIES]:
            fields += self.context.schemas[s]
            field_names += list(map(s.extract_feature_name, self.context.schemas[s]))

        mv = pmml.ModelVerification(recordCount=len(verification_input), fieldCount=len(fields))

        # step one: build verification schema
        verification_fields = pmml.VerificationFields()
        for key in fields:
            if isinstance(key, NumericFeature):
                vf = pmml.VerificationField(field=key.name, column=key.name, precision=self.EPSILON)
            else:
                vf = pmml.VerificationField(field=key.name, column=key.name)
            verification_fields.append(vf)
        mv.append(verification_fields)

        # step two: build data table
        it = pmml.InlineTable()
        for data in verification_input.iterrows():
            data = data[1]
            row = pmml.row()
            row_empty = True
            for key in field_names:
                if verification_input[key].dtype == object or not np.isnan(data[key]):
                    col = bds().createChildElement(key)
                    bds().appendTextChild(data[key], col)
                    row.append(col)
                    row_empty = False
            if not row_empty:
                it.append(row)
        mv.append(it)

        return mv

    def mining_schema(self):
        """
        Mining schema contains the model input features.
        NOTE: In order to avoid duplicates, I've decided to remove output features from MiningSchema
        NOTE: We don't need to specify any DERIVED/NUMERIC fields here, because PMML interpreter will create them
        in a lazy manner.
        """
        ms = pmml.MiningSchema()

        if Schema.INPUT in self.SCHEMAS_IN_MINING_MODEL:
            for f in sorted(self.context.schemas[Schema.INPUT], key=lambda _: _.full_name):
                ms.append(pmml.MiningField(invalidValueTreatment=f.invalid_value_treatment.value, name=f.full_name))

        for s in [Schema.OUTPUT, Schema.INTERNAL]:
            if s in self.SCHEMAS_IN_MINING_MODEL:
                for f in self.context.schemas.get(s, []):
                    ms.append(pmml.MiningField(
                        name=s.extract_feature_name(f),
                        usageType="predicted"
                    ))

        return ms

    def header(self):
        """
        Build and return Header element
        """
        return pmml.Header()

    def pmml(self, verification_data=None):
        """
        Build PMML from the context and estimator.
        Returns PMML element
        """
        p = pmml.PMML(version="4.2")
        p.append(self.header())
        p.append(self.data_dictionary())
        p.append(self.transformation_dictionary())
        p.append(self.model(verification_data))
        return p


class ClassifierConverter(EstimatorConverter):
    """
    Base class for classifier converters.
    It is required that the output schema contains only categorical features.
    The serializer will output result labels as output::feature_name and probabilities for each value of result feature
    as output::feature_name::feature_value.
    """
    def __init__(self, estimator, context):
        """
        :param estimator: Estimator to convert
        :type estimator: BaseEstimator
        :param context: context to work with
        :type context: TransformationContext
        """
        super(ClassifierConverter, self).__init__(estimator, context, ModelMode.CLASSIFICATION)
        assert isinstance(estimator, ClassifierMixin), 'Classifier converter should only be applied to the classification models'
        for f in context.schemas[Schema.OUTPUT]:
            assert isinstance(f, CategoricalFeature), 'Only categorical outputs are supported for classification task'

        # create hidden variables for each categorical output
        internal_schema = list(filter(lambda x: isinstance(x, CategoricalFeature), self.context.schemas[Schema.OUTPUT]))
        self.context.schemas[Schema.INTERNAL] = internal_schema

    def output(self):
        """
        Output section of PMML contains all model outputs.
        Classification tree output contains output variable as a label,
        and <variable>.<value> as a probability of a value for a variable
        :return: pmml.Output
        """
        output = pmml.Output()

        # the response variables
        for feature in self.context.schemas[Schema.OUTPUT]:
            output_field = pmml.OutputField(
                name=Schema.OUTPUT.extract_feature_name(feature),
                feature='predictedValue',
                optype=feature.optype.value,
                dataType=feature.data_type.value
            )
            output.append(output_field)

        # the probabilities for categories; should only be populated for classification jobs
        for feature in self.context.schemas[Schema.CATEGORIES]:
            output_field = pmml.OutputField(
                name=Schema.CATEGORIES.extract_feature_name(feature),
                optype=feature.optype.value,
                dataType=feature.data_type.value,
                feature='probability',
                targetField=Schema.INTERNAL.extract_feature_name(feature.namespace),
                value_=feature.name
            )
            output.append(output_field)

        return output


class RegressionConverter(EstimatorConverter):
    def __init__(self, estimator, context):
        super(RegressionConverter, self).__init__(estimator, context, ModelMode.REGRESSION)