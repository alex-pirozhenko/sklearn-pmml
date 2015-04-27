from sklearn_pmml import pmml
from sklearn_pmml.convert.utils import pmml_row
from sklearn_pmml.convert.features import *
from pyxb.utils.domutils import BindingDOMSupport as bds


class TransformationContext(object):
    """
    Context holder object
    """

    def __init__(self, **schemas):
        self.schemas = schemas


class EstimatorConverter(object):
    """
    A new base class for the estimator converters
    """
    PMML_VERSION = "4.2"

    MODE_CLASSIFICATION = 'classification'
    MODE_REGRESSION = 'regression'
    all_modes = {
        MODE_CLASSIFICATION,
        MODE_REGRESSION
    }

    SCHEMA_INPUT = 'input'
    SCHEMA_NUMERIC = 'numeric'
    SCHEMA_OUTPUT = 'output'

    EPSILON = 0.00001

    def __init__(self, estimator, context, mode):
        self.model_function_name = mode
        self.estimator = estimator
        self.context = context

        assert mode in self.all_modes, 'Unknown mode {}. Supported modes: {}'.format(mode, self.all_modes)

    def data_dictionary(self):
        """
        Build a data dictionary and return a DataDictionary element
        """
        dd = pmml.DataDictionary()
        for f in self.context.schemas[self.SCHEMA_INPUT] + self.context.schemas[self.SCHEMA_OUTPUT]:
            if not isinstance(f, DerivedFeature):
                data_field = pmml.DataField(dataType=f.data_type, name=f.name, optype=f.optype)
                dd.DataField.append(data_field)
                if isinstance(f, CategoricalFeature):
                    for v in f.value_list:
                        data_field.append(pmml.Value(value_=v))
        return dd

    def transformation_dictionary(self):
        """
        Build a transformation dictionary and return a TransformationDictionary element
        """
        td = pmml.TransformationDictionary()
        # define a schema with all variables available for a model
        encoded_schema = []
        self.context.schemas[self.SCHEMA_NUMERIC] = encoded_schema

        for f in self.context.schemas[self.SCHEMA_INPUT]:
            if isinstance(f, CategoricalFeature):
                ef = RealNumericFeature(name=f.name, namespace=self.SCHEMA_NUMERIC)
                # create a record in transformation dictionary with mapping from raw values into numbers
                df = pmml.DerivedField(
                    name=ef.full_name,
                    optype=ef.optype,
                    dataType=ef.data_type
                )
                mv = pmml.MapValues(outputColumn='output', dataType=ef.data_type)
                mv.append(pmml.FieldColumnPair(field=f.full_name, column='input'))
                it = pmml.InlineTable()
                for i, v in enumerate(f.value_list):
                    it.append(pmml_row(input=v, output=i))
                td.append(df.append(mv.append(it)))
            elif isinstance(f, DerivedFeature):
                ef = RealNumericFeature(name=f.name, namespace=self.SCHEMA_NUMERIC)
                df = pmml.DerivedField(
                    name=ef.full_name,
                    optype=ef.optype,
                    dataType=ef.data_type
                )
                df.append(f.transformation)
                td.append(df)
            else:
                ef = RealNumericFeature(name=f.name)

            encoded_schema.append(ef)
        assert len(encoded_schema) == len(self.context.schemas[self.SCHEMA_INPUT])
        return td

    def model(self, verification_data=None):
        """
        Build a mining model and return one of the MODEL-ELEMENTs
        """
        pass

    def model_verification(self, verification_data):
        """
        Build a model verification dataset
        :param verification_data: list of dictionaries
        :return: ModelVerification element
        """
        verification_data = pd.DataFrame(verification_data)
        fields = self.context.schemas[self.SCHEMA_INPUT] + self.context.schemas[self.SCHEMA_OUTPUT]
        assert len(verification_data) > 0, 'Verification data can not be empty'
        assert len(verification_data.columns) == len(fields), \
            'Number of fields in validation data should match to input and output schema fields'
        mv = pmml.ModelVerification(recordCount=len(verification_data), fieldCount=len(verification_data.columns))

        # step one: build verification schema
        verification_fields = pmml.VerificationFields()
        for f in fields:
            if isinstance(f, NumericFeature):
                vf = pmml.VerificationField(field=f.name, column=f.name, precision=self.EPSILON)
            else:
                vf = pmml.VerificationField(field=f.name, column=f.name)
            verification_fields.append(vf)
        mv.append(verification_fields)

        # step two: build data table
        it = pmml.InlineTable()
        for data in verification_data.iterrows():
            data = data[1]
            row = pmml.row()
            for f in fields:
                col = bds().createChildElement(f.name)
                bds().appendTextChild(data[f.name], col)
                row.append(col)
            it.append(row)
        mv.append(it)

        return mv

    def mining_schema(self):
        """
        Build a mining schema and return MiningSchema element
        """
        ms = pmml.MiningSchema()

        for f in self.context.schemas[self.SCHEMA_INPUT]:
            ms.append(pmml.MiningField(invalidValueTreatment=f.invalid_value_treatment, name=f.name))

        for f in self.context.schemas[self.SCHEMA_OUTPUT]:
            ms.append(pmml.MiningField(
                name=f.full_name,
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