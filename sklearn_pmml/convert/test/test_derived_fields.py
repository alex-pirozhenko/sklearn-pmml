import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn_pmml import EstimatorConverter, TransformationContext, pmml
from sklearn_pmml.convert import Schema, ModelMode
from sklearn_pmml.convert.features import *

test_cases = [
    (
        [
            RealNumericFeature(name='f1'),
        ],
        [
            DerivedFeature(
                feature=RealNumericFeature(name='f2'),
                transformation=pmml.Discretize(mapMissingTo=0, defaultValue=1, field='f1')
            )
        ],
        [RealNumericFeature(name='f3')],

        '<?xml version="1.0" ?>'
        '<ns1:DataDictionary xmlns:ns1="http://www.dmg.org/PMML-4_2">'
        '<ns1:DataField dataType="double" name="f1" optype="continuous"/>'
        '<ns1:DataField dataType="double" name="f3" optype="continuous"/>'
        '</ns1:DataDictionary>',

        '<?xml version="1.0" ?>'
        '<ns1:TransformationDictionary xmlns:ns1="http://www.dmg.org/PMML-4_2">'
        '<ns1:DerivedField dataType="double" name="f2" optype="continuous">'
        '<ns1:Discretize defaultValue="1" field="f1" mapMissingTo="0"/>'
        '</ns1:DerivedField>'
        '</ns1:TransformationDictionary>'
    )
]

@pytest.mark.parametrize("input_fields,derived_fields,output_fields,expected_data_dictionary,expected_transformation_dictionary", test_cases)
def test_transformation_dictionary(input_fields, derived_fields, output_fields, expected_data_dictionary, expected_transformation_dictionary):
    converter = EstimatorConverter(
        DecisionTreeClassifier(),
        context=TransformationContext({
            Schema.INPUT: input_fields,
            Schema.DERIVED: derived_fields,
            Schema.MODEL: input_fields + derived_fields,
            Schema.OUTPUT: output_fields
        }),
        mode=ModelMode.CLASSIFICATION
    )

    assert converter.data_dictionary().toxml() == expected_data_dictionary, 'Error in data dictionary generation'
    assert converter.transformation_dictionary().toxml() == expected_transformation_dictionary,\
        'Error in transformation dictionary generation'