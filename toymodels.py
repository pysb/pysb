nan = float("nan")

class Model:
    rules = []

    def __init__(self, rules):
        for r in rules:
            if r:
                self.rules += [r]


class Species:
    name = '<unnamed>'

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return '%s' % (self.name)


class Rule:
    reactants = []
    products = []


class RuleIrreversible(Rule):
    rate = nan

    def __init__(self, reactants, products, rate):
        self.reactants = reactants
        self.products = products
        self.rate = rate

    def __str__(self):
        return '%s --> %s (%s)' % (
            ' + '.join(str(r) for r in self.reactants),
            ' + '.join(str(p) for p in self.products),
            self.rate
            )


class RuleReversible(Rule):
    rates = [nan, nan]

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
