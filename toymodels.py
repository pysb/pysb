nan = float("nan")

class Rule:
    reactants = []
    products = []

class RuleIrreversible(Rule):
    rate = nan

class RuleReversible(Rule):
    rates = [nan, nan]
