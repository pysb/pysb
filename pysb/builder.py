"""
A wrapper class, ``Builder``, that facilitates the programmatic construction of
PySB models while adding a few features useful for model calibration.

The pattern for model construction using this class does not rely on the
SelfExporter class of PySB. Instead, the ``Builder`` class contains an instance
of a PySB model object. Monomers, Parameters, Rules, etc. are added to this
model object by invoking the wrapper methods of the class, which include

- :py:meth:`pysb.builder.Builder.monomer`
- :py:meth:`pysb.builder.Builder.parameter`
- :py:meth:`pysb.builder.Builder.rule`
- :py:meth:`pysb.builder.Builder.compartment`
- :py:meth:`pysb.builder.Builder.initial`
- :py:meth:`pysb.builder.Builder.observable`

Each of these functions invokes the appropriate PySB component constructor
while setting the argument ``_export=False`` so that the SelfExporter is not
invoked. The created components are then added to the instance of the model
contained in the builder class.

In addition, the model builder class implements the ``__getitem__`` method so
that invoking ``self['component_name']`` from within any method returns the
component with the given name.

In addition to managing model components, the method for parameter creation,
:py:meth:`psyb.builder.Builder.parameter`, allows the caller to specify
whether the parameter should be estimated during model calibration, and if
so, what the prior distribution should be.

Creating custom model builders
------------------------------

A useful application of the Builder class is to create subclasses that
implement macros or motifs for combinatorial model building.  For instance, a
subclass could be created as follows::

    class MyBuilder(pysb.builder.Builder):
        # Constructor with monomer declarations, etc....

        def my_motif():
            k1 = self.parameter('k1', 1, _estimate=False)
            k2 = self.parameter('k2', 1, prior=Uniform(-5, -1))
            A = self['A']
            B = self['B']
            self.rule('A_B_bind', A(b=None) + B(b=None) <> A(b=1) % B(b=1),
                      k1, k2)

This motif, implemented as a method, does several things: it manages the
addition of components to the model, keeps track of which parameters to
estimate (``k2`` but not ``k1``), and specifies the prior distribution to use
for ``k2``.

A builder subclass can thus be written containing several motifs, different
subsets of which can be called to build up models in a combinatorial
fashion. Furthermore, implementations of specific motifs can be overridden by
creating additional subclasses.
"""

from pysb import *

class Builder(object):

    # -- CONSTRUCTOR AND MONOMER DECLARATIONS --------------------------------
    def __init__(self, params_dict=None):
        """Base constructor for all model builder classes.

        Initializes collections of the parameters to estimate, as well
        as the means and variances of their priors.

        Parameters
        ----------
        params_dict : dict
            The params_dict allows any parameter value to be overriden
            by name; any parameters not included in the dict will be set
            to default values. For example, if params_dict contains::

                {'tBid_Bax_kf': 1e-2}

            then the parameter tBid_Bax_kf will be assigned a value of 1e-2;
            all other parameters will take on default values. However,
            note that the parameter value given will be multiplied by any
            scaling factor passed in when the parameter is declared.
        """

        self.model = Model('model', _export=False)
        """The PySB model to which components are added."""
        self.estimate_params = []
        """A list of the parameters to be estimated."""
        self.params_dict = params_dict
        """A dict of parameter values to override default values."""
        self.priors = []
        """A list of priors for use in sampling parameter values."""

    # -- CONSTRUCTOR WRAPPER FUNCTIONS ---------------------------------------
    def monomer(self, *args, **kwargs):
        """Adds a parameter to the Builder's model instance."""
        m = Monomer(*args, _export=False, **kwargs)
        self.model.add_component(m)
        return m

    def parameter(self, name, value, factor=1, prior=None):
        """Adds a parameter to the Builder's model instance.

        Examines the params_dict attribute of the Builder instance (which is
        set in the constructor,
        :py:meth:`pysb.builder.Builder.__init__`).  If the
        parameter with the given name is in the ``params_dict``, then the value
        in the ``params_dict`` is used to construct the parameter, and the
        argument ``value`` is ignored. If the parameter is not in
        ``params_dict``, then the parameter is assigned ``value``.

        Furthermore, in all cases the parameter value is multiplied by a
        scaling factor specified by the argument ``factor``. This allows rate
        scaling factor that are dependent on model implementations or on units
        (deterministic vs. stochastic) to be handled by keeping the same
        parameter value but passing in the appropriate value for ``factor``.

        Parameters
        ----------
        name : string
            The name of the parameter to add
        value : number
            The value of the parameter
        factor : number
            A scaling factor to be applied to the parameter value.
        prior : instance of prior class from bayessb.priors
            The prior object describing the prior probability of different
            values for this parameter, if the parameter should be included among
            the parameters to estimate (contained in the set
            ``Builder.estimate_params``).
        """

        if self.params_dict is None:
            param_val = value * factor
        else:
            if name in self.params_dict:
                param_val = self.params_dict[name] * factor
            else:
                param_val = value * factor

        p = Parameter(name, param_val, _export=False)
        self.model.add_component(p)

        if prior is not None:
            self.estimate_params.append(p)
            self.priors.append(prior)

        return p

    def rule(self, *args, **kwargs):
        """Adds a rule to the Builder's model instance."""
        r = Rule(*args, _export=False, **kwargs)
        self.model.add_component(r)
        return r 

    def compartment(self, *args, **kwargs):
        """Adds a compartment to the Builder's model instance."""
        c = Compartment(*args, _export=False, **kwargs)
        self.model.add_component(c)
        return c 

    def observable(self, *args, **kwargs):
        """Adds an observable to the Builder's model instance."""
        o = Observable(*args, _export=False, **kwargs)
        self.model.add_component(o)
        return o

    def expression(self, *args, **kwargs):
        """Adds an expression to the Builder's model instance."""
        e = Expression(*args, _export=False, **kwargs)
        self.model.add_component(e)
        return e

    def initial(self, *args):
        """Adds an initial condition to the Builder's model instance."""
        self.model.initial(*args)

    def __getitem__(self, index):
        """Returns the component with the given string index
        from the instance of the model contained by the Builder."""
        return self.model.all_components()[index]


