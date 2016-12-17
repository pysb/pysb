from __future__ import print_function
import pysb
import abc
from pysb.core import SelfExporter
from pysb.pattern import SpeciesPatternMatcher, ReactionPatternMatcher, \
    RulePatternMatcher


class ModelAssertionFailure(Exception):
    def __init__(self, assertion, model, message=None):
        self.assertion = assertion
        self.model = model
        self.message = message

    def __repr__(self):
        base_msg = '%s on model %s failed' % (self.assertion,
                                              self.model.name)
        if self.message:
            return base_msg + ': ' + str(self.message)
        else:
            return base_msg

    def __str__(self):
        return repr(self)


class TestSuite(object):
    """
    A suite of tests for checking properties of a model

    There are two modes of operation: building a test suite using add() and
    executing all the tests at once with check_all(), or executing tests
    immediately with check().

    Examples
    --------

    Create a test suite for the EARM 1.0 model:

    >>> from pysb.testing.modeltests import TestSuite, SpeciesExists, \
        SpeciesDoesNotExist
    >>> from pysb.bng import generate_equations
    >>> from pysb.examples.earm_1_0 import model
    >>> ts = TestSuite(model)

    Create variables for model components (not needed for models defined
    interactively):

    >>> AMito, mCytoC, mSmac, cSmac, L, CPARP = [model.monomers[m] for m in \
                                       ('AMito', 'mCytoC', 'mSmac', 'cSmac', \
                                        'L', 'CPARP')]

    Add some assertions:

    Check that AMito(b=1) % mSmac(b=1) exists in the species graph (note this
    doesn't guarantee the species will actually be producted/consumed/change
    in concentration; that depends on the rate constants):

    >>> ts.add(SpeciesExists(AMito(b=1) % mSmac(b=1)))

    This is the opposite check, that the complex above doesn't exist,
    which should of course fail:

    >>> ts.add(SpeciesDoesNotExist(AMito(b=1) % mSmac(b=1)))

    We can also specify that species matching a pattern should never exist
    in a model. For example, we shouldn't ever be producing unbound
    ligand in the EARM 1.0 model:

    >>> ts.add(SpeciesNeverProduct(L(b=None)))

    We could also have used SpeciesOnlyReactant. The difference is the
    latter checks for an appearance as a reactant, whereas
    SpeciesNeverProduct would pass whether the species appeared as a
    reactant or not.

    >>> ts.add(SpeciesOnlyReactant(L(b=None)))

    CPARP is an output in this model, so it should appear as a product but
    never as a reactant:

    >>> ts.add(SpeciesOnlyProduct(CPARP()))

    When we're ready, we can generate the reactions and check the assertions:

    >>> generate_equations(model)
    >>> ts.check_all()  # doctest:+ELLIPSIS
    SpeciesExists(AMito() % mSmac())...OK...
    SpeciesDoesNotExist(AMito() % mSmac())...FAIL...
      [AMito(b=1) % mSmac(b=1)]...
    SpeciesExists(AMito(b=1) % mCytoC(b=1))...OK...
    SpeciesNeverProduct(L(b=None))...OK...
    SpeciesOnlyProduct(CPARP())...OK...

    We can also execute any test immediately without adding it to the test
    suite (note that some tests require a reaction network to be generated):

    >>> ts.check(SpeciesExists(L(b=None)))
    True
    """
    _KNOWN_CACHES = {'species_pattern_matcher': SpeciesPatternMatcher,
                     'rule_pattern_matcher': RulePatternMatcher,
                     'reaction_pattern_matcher': ReactionPatternMatcher}
    _COL = {'OK': '\033[92m', 'FAIL': '\033[91m', 'END': '\033[0m'}

    def __init__(self, model=None):
        self._caches = {}
        self.assertions = []
        self._model = model
        if model:
            self._model = model
        elif SelfExporter.default_model:
            self._model = SelfExporter.default_model
        else:
            raise Exception('A model must be specified explicitly if the '
                            'PySB self-exporter is not in use')

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        """ Changing the model invalidates any caches """
        self._caches = {}
        self._model = model

    def _ensure_required_caches(self, assertion):
        for cache in assertion.required_caches:
            if cache not in self._KNOWN_CACHES.keys():
                raise Exception('Unknown assertion cache: %s' % cache)
            self._caches[cache] = self._KNOWN_CACHES[cache](self.model)

    def add(self, assertion):
        self.assertions.append(assertion)

    def check(self, assertion):
        """
        Checks an assertion immediately without adding it to the test suite

        Parameters
        ----------
        assertion: ModelAssertion
            An instance of the ModelAssertion subclass

        Returns
        -------
        True if assertion succeeded or raises a ModelAssertionFailure
        exception if not
        """

        self._ensure_required_caches(assertion)
        return assertion.check(self.model, **{name: self._caches[name]
                                              for name in
                                              assertion.required_caches})

    def check_all(self, stop_on_exception=False):
        """Runs all assertions in the test suite"""
        for a in self.assertions:
            print('%s... ' % repr(a), end="")
            try:
                self.check(a)
                print('%sOK%s' % (self._COL['OK'], self._COL['END']))
            except ModelAssertionFailure as e:
                print('%sFAIL%s' % (self._COL['FAIL'], self._COL['END']))
                print('  ' + str(e.message))
                if stop_on_exception:
                    return
            except Exception as e:
                print('%sERROR%s' % (self._COL['FAIL'], self._COL['END']))
                print('  ' + str(e))
                if stop_on_exception:
                    return


class ModelAssertion(object):
    """
    Base class for model assertions
    """
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        self.required_caches = set()
        self._last_result = None

    def __repr__(self):
        return '%s()' % self.__class__.__name__

    @abc.abstractmethod
    def check(self, model, **kwargs):
        if not isinstance(model, pysb.Model):
            raise ValueError('model should be an instance of pysb.Model')


def _negated_subclass(assertion_class, class_name):
    """
    Creates a negated version of an assertion class through subclassing

    Parameters
    ----------
    assertion_class: class
        ModelAssertion or its subclasses (not an instance)

    Returns
    -------
    A negated version of the ModelAssertion class

    """
    class NegatedSubclass(assertion_class):
        """
        Negated version of :class:`.{}`
        """.format(assertion_class.__class__.__name__)

        def check(self, model, **kwargs):
            try:
                super(self.__class__, self).check(model, **kwargs)
            except ModelAssertionFailure:
                return True

            raise ModelAssertionFailure(assertion=self, model=model,
                                        message=self._last_result)

    NegatedSubclass.__name__ = class_name
    return NegatedSubclass


class ReactionNetworkAssertion(ModelAssertion):
    """
    Base class for reaction network assertions

    Checks the reaction network has been generated
    """
    def __init__(self, *args, **kwargs):
        super(ReactionNetworkAssertion, self).__init__(*args, **kwargs)

    @abc.abstractmethod
    def check(self, model, **kwargs):
        super(ReactionNetworkAssertion, self).check(model)
        if not model.species:
            raise ModelAssertionFailure(assertion=self,
                                        model=model,
                                        message='Reaction network has not '
                                                'been generated yet')


class SpeciesAssertion(ReactionNetworkAssertion):
    """
    Class for checking species within a reaction network
    """
    def __init__(self, *args, **kwargs):
        super(SpeciesAssertion, self).__init__(*args, **kwargs)
        self.required_caches.add('species_pattern_matcher')
        self.pattern = args[0]

    @abc.abstractmethod
    def check(self, model, **kwargs):
        super(SpeciesAssertion, self).check(model)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.pattern)


class SpeciesExists(SpeciesAssertion):
    """
    Checks a species pattern exists in the list of species
    """
    def check(self, model, **kwargs):
        self._last_result = kwargs['species_pattern_matcher'].match(
            self.pattern)
        if self._last_result:
            return True
        else:
            raise ModelAssertionFailure(assertion=self, model=model)


SpeciesDoesNotExist = _negated_subclass(SpeciesExists, 'SpeciesDoesNotExist')


class ReactionAssertion(ReactionNetworkAssertion):
    def __init__(self, *args, **kwargs):
        super(ReactionAssertion, self).__init__(*args, **kwargs)
        self.required_caches.add('reaction_pattern_matcher')
        self.pattern = args[0]

    @abc.abstractmethod
    def check(self, model, **kwargs):
        super(ReactionAssertion, self).check(model)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.pattern)


class SpeciesIsProduct(ReactionAssertion):
    """
    Checks a species pattern appears on the product side of a reaction
    """
    def check(self, model, **kwargs):
        self._last_result = kwargs[
            'reaction_pattern_matcher'].match_products(self.pattern)
        if not self._last_result:
            raise ModelAssertionFailure(assertion=self, model=model)
        else:
            return True

SpeciesNeverProduct = _negated_subclass(SpeciesIsProduct, 'SpeciesNeverProduct')


class SpeciesIsReactant(ReactionAssertion):
    """
    Checks a species pattern appears on the reactant side of a reaction
    """
    def check(self, model, **kwargs):
        self._last_result = kwargs[
            'reaction_pattern_matcher'].match_reactants(self.pattern)
        if self._last_result:
            raise ModelAssertionFailure(assertion=self, model=model,
                                        message=self._last_result)
        else:
            return True

SpeciesNeverReactant = _negated_subclass(SpeciesIsReactant,
                                         'SpeciesNeverReactant')


class SpeciesOnlyProduct(ReactionAssertion):
    """ Checks a species appears as a product but never as a reactant """
    def check(self, model, **kwargs):
        rpm = kwargs['reaction_pattern_matcher']
        if not rpm.match_products(self.pattern):
            raise ModelAssertionFailure(assertion=self,
                                        model=model,
                                        message='Does not appear as product')
        as_reactant = rpm.match_reactants(self.pattern)
        if as_reactant:
            raise ModelAssertionFailure(assertion=self,
                                        model=model,
                                        message='Appears as reactant:' +
                                                str(as_reactant))

        return True


class SpeciesOnlyReactant(ReactionAssertion):
    """ Checks a species appears as a reactant but never as a product """

    def check(self, model, **kwargs):
        rpm = kwargs['reaction_pattern_matcher']
        if not rpm.match_reactants(self.pattern):
            raise ModelAssertionFailure(assertion=self,
                                        model=model,
                                        message='Does not appear as reactant')
        as_product = rpm.match_products(self.pattern)
        if as_product:
            raise ModelAssertionFailure(assertion=self,
                                        model=model,
                                        message='Appears as product: ' +
                                                str(as_product))

        return True


class RuleAssertion(ModelAssertion):
    def __init__(self, *args, **kwargs):
        super(RuleAssertion, self).__init__(*args, **kwargs)
        self.required_caches.add('rule_pattern_matcher')
        self.pattern = args[0]

    @abc.abstractmethod
    def check(self, model, **kwargs):
        super(RuleAssertion, self).check(model)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.pattern)


class AllObservablesInRules(RuleAssertion):
    def check(self, model, **kwargs):
        rpm = kwargs['rule_pattern_matcher']
        unmatched_observables = []
        for obs in model.observables:
            matches = rpm.match_rules(obs.reaction_pattern)
            if not matches:
                unmatched_observables.append(obs)
        if unmatched_observables:
            raise ModelAssertionFailure(assertion=self,
                                        model=model,
                                        message='Unmatched Observables: ' +
                                                str(unmatched_observables))

        return True
