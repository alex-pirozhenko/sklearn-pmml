estimator_to_converter = {}


def find_converter(estimator):
    # TODO: do the search here
    return estimator_to_converter.get(estimator.__class__, None)