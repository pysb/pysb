nan = float("nan")

class Model:
    def __init__(self, rules):
        self.rules = []
        self.species = {}
        for r in rules:
            if r:
                self.rules.append(r)

    def finalize(self):
        for r in self.rules:
            new_rules = [r]
            if r.__class__ is RuleReversible:
                self.rules.remove(r)
                self.rules += [
                    RuleIrreversible(reactants=r.reactants, products=r.products, rate=r.rates[0]),
                    RuleIrreversible(reactants=r.products, products=r.reactants, rate=r.rates[1]),
                    ]
        for r in self.rules:
            r.coalesce_species(self.species)
            r.generate_mass_action_terms()


class Species:
    def __init__(self, name):
        self.name = name
        self.mass_action_terms = []

    def __str__(self):
        return '%s' % (self.name)


class Rule:
    def coalesce_species(self, species):
        for sl in [self.reactants, self.products]:
            for i in range(len(sl)):
                s = sl[i]
                if s.name in species:
                    s = species[s.name]
                    sl[i] = s
                else:
                    species[s.name] = s


class RuleIrreversible(Rule):
    def __init__(self, reactants, products, rate):
        self.reactants = reactants
        self.products = products
        self.rate = rate

    def __str__(self):
        return '%s --> %s (%s)' % (
            ' + '.join(r.name for r in self.reactants),
            ' + '.join(p.name for p in self.products),
            self.rate
            )

    def generate_mass_action_terms(self):
        for t in [(self.products, self.reactants, 1), (self.reactants, self.reactants, -1)]:
            for s in t[0]:
                s.mass_action_terms.append(MassActionTerm(species=t[1], factor=t[2]*self.rate))


class RuleReversible(Rule):
    def __init__(self, reactants, products, rates):
        self.reactants = reactants
        self.products = products
        self.rates = rates

    def __str__(self):
        return '%s <-> %s (%s)' % (
            ' + '.join(str(r) for r in self.reactants),
            ' + '.join(str(p) for p in self.products),
            ', '.join(str(r) for r in self.rates)
            )


class MassActionTerm:
    def __init__(self, species, factor):
        self.species = species
        self.factor = factor

    def __str__(self):
        return '%s*%s' % (
            self.factor,
            ''.join('['+str(s)+']' for s in self.species)
            )
