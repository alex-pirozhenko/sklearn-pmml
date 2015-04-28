from functools import partial
from sklearn_pmml import pmml
from pyxb.utils.domutils import BindingDOMSupport as bds
import numpy as np
estimator_to_converter = {}


def find_converter(estimator):
    # TODO: do the search here
    return estimator_to_converter.get(estimator.__class__, None)


def pmml_row(**columns):
    """
    Creates pmml.row element with columns
    :param columns: key-value pairs to be inserted into the row
    :return: pmml.row element
    """
    r = pmml.row()
    for name, value in columns.items():
        el = bds().createChildElement(name)
        bds().appendTextChild(value, el)
        r.append(el)
    return r


class DerivedFeatureTransformations(object):
    """
    A helper for building Derived Feature transformations. Creates both transformation and the DerivedFeature content.
    Typical usage of the methods:

    DerivedFeature(
            RealNumericFeature('my_derived_feature'),
            **DerivedFeatureTransformations.field_in_list('input_feature', ['A', 'B', 'C'])
    )
    """
    TRANSFORMATION = 'transformation'
    FUNCTION = 'function'

    @staticmethod
    def field_in_list(field, values):
        mv = pmml.MapValues(outputColumn='output', defaultValue=0)
        mv.append(pmml.FieldColumnPair(field=field, column='input'))
        it = pmml.InlineTable()
        for v in values:
            it.append(pmml_row(input=v, output=1))
        mv.append(it)
        return {
            DerivedFeatureTransformations.TRANSFORMATION: mv,
            DerivedFeatureTransformations.FUNCTION: lambda df: reduce(np.logical_or, [df[field] == _ for _ in values])
        }

    @staticmethod
    def field_not_in_list(field, values):
        mv = pmml.MapValues(outputColumn='output', defaultValue=1)
        mv.append(pmml.FieldColumnPair(field=field, column='input'))
        it = pmml.InlineTable()
        for v in values:
            it.append(pmml_row(input=v, output=0))
        mv.append(it)
        return {
            DerivedFeatureTransformations.TRANSFORMATION: mv,
            DerivedFeatureTransformations.FUNCTION: lambda df: reduce(np.logical_and, [df[field] != _ for _ in values])
        }

    @staticmethod
    def map_values(field, value_map, default_value):
        mv = pmml.MapValues(outputColumn='output', default_value=default_value)
        mv.append(pmml.FieldColumnPair(field=field, column='input'))
        it = pmml.InlineTable()
        for k, v in value_map.items():
            it.append(pmml_row(input=k, output=v))
        mv.append(it)
        return {
            DerivedFeatureTransformations.TRANSFORMATION: mv,
            DerivedFeatureTransformations.FUNCTION:
                lambda df: np.vectorize(partial(value_map.get, default_value))(df[field])
        }

    @staticmethod
    def arithmetics(tree):
        """
        Takes an arithmetic operations tree (Lisp-styled) as an input
        """

        def basic_function(func_name, args):
            expr = pmml.Apply(function=func_name)
            for a in args:
                expr.append(a)
            return expr

        def mod_function(args):
            expr = pmml.Apply(function='-')
            expr.append(args[0])
            mul = pmml.Apply(function='*')
            mul.append(args[1])
            floor = pmml.Apply(function='floor')
            mul.append(floor)
            div = pmml.Apply(function='/')
            floor.append(div)
            div.append(args[0])
            div.append(args[1])
            return expr

        # TODO: test me
        def greedy_evaluation(node):
            if isinstance(node, str):
                # field reference
                return (lambda df: df[node]), pmml.FieldRef(field=node)
            elif isinstance(node, (tuple, list)):
                # eval arguments
                args = map(greedy_evaluation, node[1:])
                functions = {
                    '*': lambda df: np.multiply(*[_[0](df) for _ in args]),
                    '-': lambda df: np.subtract(*[_[0](df) for _ in args]),
                    '+': lambda df: np.add(*[_[0](df) for _ in args]),
                    '/': lambda df: np.divide(*[_[0](df) for _ in args]),
                    '%': lambda df: np.mod(*[_[0](df) for _ in args]),
                }
                assert isinstance(node[0], str), 'First element should be a code of operation'
                assert node[0] in functions, 'Unknown function code {}. Supported codes: {}'.format(node[0], functions.keys())
                expr = {
                    '*': partial(basic_function, '*'),
                    '-': partial(basic_function, '-'),
                    '+': partial(basic_function, '+'),
                    '/': partial(basic_function, '/'),
                    '%': mod_function
                }.get(node[0])([a[1] for a in args])
                func = functions[node[0]]
                return func, expr
            else:
                # numeric terminal
                return lambda df: node, pmml.Constant(node, dataType='double')

        function, transformation = greedy_evaluation(tree)

        return {
            DerivedFeatureTransformations.TRANSFORMATION: transformation,
            DerivedFeatureTransformations.FUNCTION: function
        }

    @staticmethod
    def replace_value(field, original, replacement):
        if original is not None:
            transformation = pmml.Apply(function='if')
            cond = pmml.Apply(function='equals')
            cond.append(pmml.FieldRef(field=field))
            cond.append(pmml.Constant(original))
            transformation.append(pmml.Constant(replacement))
            transformation.append(pmml.FieldRef(field=field))

            return {
                DerivedFeatureTransformations.TRANSFORMATION: transformation,
                DerivedFeatureTransformations.FUNCTION: lambda df: np.where(df[field] == original, replacement, df[field])
            }
        else:
            transformation = pmml.Apply(function='+', mapMissingTo=replacement)
            transformation.append(pmml.Constant(0))
            transformation.append(pmml.FieldRef(field=field))
            return {
                DerivedFeatureTransformations.TRANSFORMATION: transformation,
                DerivedFeatureTransformations.FUNCTION: lambda df: np.where(df[field].isnull(), replacement, df[field])
            }

