from sklearn_pmml import pmml
from pyxb.utils.domutils import BindingDOMSupport as bds
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