from sklearn_pmml import pmml
from sklearn_pmml.convert.features import Feature, NumericFeature, CategoricalFeature
from sklearn_pmml.pmml import MiningField
from pyxb.utils.domutils import BindingDOMSupport as bds

__all__ = ['Schema', 'PMMLTransformationContext', 'PMMLBuilder']

estimator_to_converter = {}

class Converter(object):
    """
    A base class for the converters.
    """
    def transform(self, obj, ctx):
        """
        Serialize object to PMML node
        :param obj: object to transform
        :return PMML node
        """
        if not self.is_applicable(obj, ctx):
            raise ValueError('{} can not process {}'.format(
                self.__class__.__name__, obj
            ))

    def is_applicable(self, obj, ctx):
        """
        Check whether the converter can process object under provided context
        """
        raise NotImplementedError()

    @staticmethod
    def mining_schema(schema):
        """
        Prepare mining schema node
        """
        ms = pmml.MiningSchema()
        for f in schema.features:
            assert isinstance(f, Feature)
            ms.append(pmml.MiningField(
                invalidValueTreatment=f.invalid_value_treatment,
                name=f.external_name
            ))
        return ms


class Schema(object):
    def __init__(self, features, output):
        self._features = features
        self._output = output

    @property
    def features(self):
        return self._features

    @property
    def output(self):
        return self._output


class PMMLTransformationContext(object):
    """
    Context for sklearn -> PMML transformations
    """
    def __init__(self, schema, metadata):
        assert isinstance(schema, Schema)
        self.schema = schema
        self.metadata = metadata


class PMMLBuilder(object):
    def build(self, obj, ctx):
        converter = self.find_converter(obj.__class__)
        assert converter is not None, "Can not find converter for {}".format(obj)
        p = pmml.PMML(version="4.2")
        p.append(pmml.Header(**ctx.metadata))
        for el in self.data_description(obj, ctx):
            p.append(el)
        p.append(converter.transform(obj, ctx))
        return p

    def data_description(self, obj, ctx):
        dd = pmml.DataDictionary()
        td = pmml.TransformationDictionary()
        for f in ctx.schema.features:

            data_field = pmml.DataField(dataType=f.data_type, name=f.external_name, optype=f.optype)
            dd.DataField.append(data_field)
            if isinstance(f, CategoricalFeature):
                df = pmml.DerivedField(
                    name=f.internal_name,
                    optype="continuous",
                    dataType="integer"
                )
                mv = pmml.MapValues(outputColumn='output', dataType='integer')
                mv.append(pmml.FieldColumnPair(field=f.external_name, column='input'))
                it = pmml.InlineTable()
                for i, v in enumerate(f.value_list):
                    r = pmml.row()
                    input = bds().createChildElement('input')
                    bds().appendTextChild(v, input)
                    output = bds().createChildElement('output')
                    bds().appendTextChild(i, output)
                    r.append(input)
                    r.append(output)
                    it.append(r)
                    data_field.append(pmml.Value(value_=v))

                td.append(df.append(mv.append(it)))
        return dd, td

    @staticmethod
    def find_converter(cls):
        # TODO: do the search here
        return estimator_to_converter.get(cls, None)
