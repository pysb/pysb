import pysb, pysb.bng, warnings, re, sympy


class JacobianGenerator(object):

    def __init__(self, model):
        self.model = model
        self.indent_level = 0
        self.__content = None

    def get_content(self, sim_length=1):
        if self.__content == None:
            self.generate_content(sim_length=sim_length)
        assert self.indent_level == 0, "missing %d outdent call(s)" % self.indent_level
        return self.__content

    def emit(self, text=''):
        self.__content += self.indent_level * 2 * ' ' + text + "\n"

    def indent(self):
        self.indent_level += 1

    def outdent(self):
        assert self.indent_level > 0, "called outdent without matching indent"
        self.indent_level -= 1

    def generate_content(self, sim_length):
        pysb.bng.generate_equations(self.model)
        self.__content = ''
        self.emit("DECLARE")
        self.indent()
        self.generate_types()
        self.outdent()
        self.emit("END")
        self.emit()
        self.emit("MODEL %s" % self.model.name)
        self.indent()
        self.generate_parameters()
        self.generate_variables()
        self.generate_equations()
        self.outdent()
        self.emit("END")
        self.emit()
        self.emit("SIMULATION SIM")
        self.indent()
        self.generate_options()
        self.generate_units()
        self.generate_set()
        self.generate_initial()
        self.generate_schedule(sim_length)
        self.outdent()
        self.emit("END")

    def generate_types(self):
        self.emit("TYPE");
        self.indent()
        self.emit("NOTYPE = 0.0 : -1E300 : 1E300")
        self.outdent()

    def generate_parameters(self):
        self.emit("PARAMETER")
        self.indent()
        #for param_group in self.make_groups(self.model.parameters, 5):
        for param_group in self.make_groups([p for p in self.model.parameters if p.name[-2:] != '_0'], 5):
            self.emit(', '.join([p.name for p in param_group]) + ' AS REAL')
        self.outdent()

    def generate_variables(self):
        self.emit("VARIABLE")
        self.indent()
        obs_names = self.model.observable_groups.keys()
        for obs_group in self.make_groups(obs_names, 5):
            self.emit(', '.join(obs_group) + ' AS NOTYPE')
        var_names = ['s%d' % i for i in range(len(self.model.species))]
        for var_group in self.make_groups(var_names, 5):
            self.emit(', '.join(var_group) + ' AS NOTYPE')
        self.outdent()

    def generate_equations(self):
        self.emit("EQUATION")
        self.indent()
        obs_names = self.model.observable_groups.keys()
        obs_exprs = [' + '.join('%g * s%s' % g for g in self.model.observable_groups[name]) for name in obs_names]
        for obs in zip(obs_names, obs_exprs):
            self.emit('%s = %s;' % obs)
        var_names = ['s%d' % i for i in range(len(self.model.species))]
        for (name, ode) in zip(var_names, self.model.odes):
            expression = sympy.sstr(ode)
            expression = re.sub(r'\*\*', '^', expression)
            self.emit('$%s = %s;' % (name, expression))
        self.outdent()

    def generate_options(self):
        self.emit("OPTIONS")
        self.indent()
        self.outdent()

    def generate_units(self):
        self.emit("UNIT")
        self.indent()
        self.emit("M AS " + self.model.name)
        self.outdent()

    def generate_set(self):
        self.emit("SET")
        self.indent()
        for p in self.model.parameters:
            #self.emit("M.%s := %g;" % (p.name, p.value))
            if p.name[-2:] != '_0':
                self.emit("M.%s := %g;" % (p.name, p.value))
        self.outdent()

    def generate_initial(self):
        self.emit("INITIAL")
        self.indent()
        vars_unseen = set(['s%d' % i for i in range(len(self.model.species))])
        for (pattern, param) in self.model.initial_conditions:
            sname = 's%d' % self.model.get_species_index(pattern)
            vars_unseen.remove(sname)
            #self.emit("M.%s = M.%s;" % (sname, param.name))
            self.emit("M.%s = %g;" % (sname, param.value))
        for sname in sorted(vars_unseen):
            self.emit("M.%s = 0;" % sname)
        self.outdent()

    def generate_schedule(self, sim_length):
        self.emit("SCHEDULE")
        self.indent()
        self.emit("CONTINUE FOR %d" % sim_length);
        self.outdent()

    def make_groups(self, elements, size):
        groups = []
        offsets = list(range(0, len(elements), size)) + [None]
        for i in range(0, len(offsets) - 1):
            groups.append(elements[offsets[i]:offsets[i+1]])
        return groups
