nan = float("nan")

class Model:
    def __init__(self, rules):
        self.rules = []
        self.species = {}

        for r in rules:
            if r:
                r.coalesce_species(self.species)
                new_rules = [r]
                if r.__class__ is RuleReversible:
                    new_rules = [
                        RuleIrreversible(reactants=r.reactants, products=r.products, rate=r.rates[0]),
                        RuleIrreversible(reactants=r.products, products=r.reactants, rate=r.rates[1]),
                        ]
                self.rules += new_rules


class Species:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return '%s' % (self.name)


class Rule:
    def coalesce_species(self, species):
        for sl in [self.reactants, self.products]:
            for i in range(len(sl)):
                name = sl[i]
                if name in species:
                    s = species[name]
                else:
                    s = Species(name)
                    species[name] = s
                sl[i] = s


class RuleIrreversible(Rule):
    def __init__(self, reactants, products, rate):
        self.reactants = reactants
        self.products = products
        self.rate = rate

    def __str__(self):
        return '%s --> %s (%s)' % (
            ' + '.join(r.id for r in self.reactants),
            ' + '.join(p.id for p in self.products),
            self.rate
            )


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
