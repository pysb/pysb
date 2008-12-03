from ply import yacc
from toymodels import RuleReversible, RuleIrreversible


# Get the token map from the lexer.  This is required.
from toylex import tokens


#def p_expression_species(p):
#    'expression : SPECIES'
#    print "expression_species:", p[1]
#    p[0] = [p[1]]

#def p_expression_plus(p):
#    'expression : expression PLUS SPECIES'
#    print "expression_plus:", p[1], "+", p[3]
#    p[0] = p[1] + [p[3]]

#def p_irr_rule(p):
#    'irr_rule : expression IRRARROW expression LPAREN FLOAT RPAREN'
#    print "irr_rule"
#    p[0] = RuleIrreversible(reactants=p[1], products=p[3], rate=p[5])

#def p_rev_rule(p):
#    'rev_rule : expression REVARROW expression LPAREN FLOAT COMMA FLOAT RPAREN'
#    print "rev_rule"
#    p[0] = RuleReversible(reactants=p[1], products=p[3], rates=[p[5], p[7]])

def p_statement_empty(p):
    'statement : NEWLINE'
    print "statement_empty"

#def p_statement(p):
#    'statement : expression NEWLINE'
#    print "statement:" + p[1]

def p_statement_list_trivial(p):
    '''statement_list : statement'''
    print "statement_list"

def p_statement_list(p):
    '''statement_list : statement_list statement'''
    print "statement_list"

def p_model(p):
    'model : statement_list'
    print "model"

# Error rule for syntax errors
def p_error(p):
    print "Syntax error in input:"
    print p

precedence = (
    ('left', 'PLUS'),
)

# Build the parser
yacc.yacc(write_tables=0)
