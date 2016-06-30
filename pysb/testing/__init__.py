from nose.tools import *
from pysb.core import Model, SelfExporter
import pickle

def with_model(func):
    """Decorate a test to set up and tear down a Model."""
    def inner(*args, **kwargs):
        model = Model(func.__name__, _export=False)
        # manually set up SelfExporter, targeting func's globals
        selfexporter_state = SelfExporter.do_export
        SelfExporter.do_export = True
        SelfExporter.default_model = model
        SelfExporter.target_module = func.__module__
        SelfExporter.target_globals = func.__globals__
        SelfExporter.target_globals['model'] = model
        try:
            # call the actual test function
            func(*args, **kwargs)
        finally:
            # clean up the globals
            SelfExporter.cleanup()
            SelfExporter.do_export = selfexporter_state
    return make_decorator(func)(inner)

def serialize_component_list(model, filename):
    """Serialize (pickle) the components of the given model to a file. This can
    later be used to compare the state of the model against a previously
    validated state using :py:func:`check_model_against_component_list`.
    """

    f = open(filename, 'w')
    pickle.dump(list(model.all_components().values()), f)
    f.close()

def check_model_against_component_list(model, component_list):
    """Check the components of the given model against the provided list
    of components, asserting that they are equal. Useful for testing a
    model against a previously validated (and serialized) state.

    Currently checks equality by performing a string comparison of the
    repr() of each component, however, this may be revised to use alternative
    measures of equality in the future.
    
    To serialize the list of components to create a record of a
    validated state, see :py:func:`serialize_component_list`.
    """
    assert len(model.all_components()) == len(component_list), \
           "Model %s does not have the same " \
           "number of components as the previously validated version. " \
           "The validated model has %d components, current model has " \
           "%d components." % \
           (model.name, len(model.all_components()), len(component_list))

    model_components = list(model.all_components().values())
    for i, comp in enumerate(component_list):
        model_comp_str = repr(model_components[i])
        comp_str = repr(comp) 
        assert comp_str == model_comp_str, \
               "Model %s does not match reference version: " \
               "Mismatch at component %d: %s in the reference model not " \
               "equal to %s in the current model." \
                % (model.name, i, comp_str, model_comp_str)

    assert True
