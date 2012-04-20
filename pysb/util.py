from pysb import ComponentSet

__all__ = ['rules_using_parameter']

def rules_using_parameter(model, parameter):
    """Return a ComponentSet of rules in the model which make use of the given parameter"""
    cset = ComponentSet()
    for rule in model.rules:
        if rule.rate_forward is parameter or rule.rate_reverse is parameter:
            cset.add(rule)
    return cset
