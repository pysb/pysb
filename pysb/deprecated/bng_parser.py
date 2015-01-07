from __future__ import print_function
from ply import lex, yacc


reserved_list = [
    'begin',
    'end',
    'parameters',
    'molecule_types',
    'species',
    'reaction_rules',
    'observables',
    ]
reserved = dict((r, r.upper()) for r in reserved_list)

tokens = [
    'ID',
    
    'FLOAT',
    'INTEGER',

    'COMMA',
    'PLUS',
    'TILDE',
    'EXCLAMATION',
    'QUESTION',
    'PERIOD',
    'IRRARROW',
    'REVARROW',
    'LPAREN',
    'RPAREN',
    'NEWLINE',
    ] + list(reserved.values())

t_COMMA       = r','
t_PLUS        = r'\+'
t_TILDE       = r'~'
t_EXCLAMATION = '!'
t_QUESTION    = '\?'
t_PERIOD      = '\.'
t_IRRARROW    = r'-->'
t_REVARROW    = r'<->'
t_LPAREN      = r'\('
t_RPAREN      = r'\)'

# Define a rule so we can track line numbers
def t_NEWLINE(t):
    r'\n'
    t.lexer.lineno += 1
    return t

def t_FLOAT(t):
    r'[+-]?(\d*\.\d+([eE][+-]?\d+)?|\d+[eE][+-]?\d+)'
    try:
        t.value = float(t.value)    
    except ValueError:
        print("Line %d: Number '%s' has some kind of problem (ValueError)!" % (t.lineno,t.value))
        t.value = float("nan")
    return t

def t_INTEGER(t):
    r'\d+'
    try:
        t.value = int(t.value)    
    except ValueError:
        print("Line %d: Number '%s' has some kind of problem (ValueError)!" % (t.lineno,t.value))
        t.value = 0
    return t

def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value,'ID')  # check for reserved words
    return t

# Match and ignore comments (# to end of line)
def t_comment(t):
    r'\#[^\n]*'

# A string containing ignored characters (spaces and tabs)
t_ignore  = ' \t'

# Error handling rule
def t_error(t):
    print("Illegal character '%s' on line %d" % (t.value[0], t.lineno))
    t.lexer.skip(1)



#from toymodels import Model, Species, RuleReversible, RuleIrreversible


def list_helper(p):
    if len(p) == 1:
        p[0] = []
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = p[1] + [p[2]]
    p[0] = [v for v in p[0] if v != None] # filter out Nones


def p_model(p):
    'model : block_list'
    p[0] = p[1]
    print("model:", p[0])

def p_block_list(p):
    '''block_list : block_list block
                  | block'''
    list_helper(p)

def p_block(p):
    '''block : parameter_block
             | molecule_type_block
             | species_block
             | reaction_rules_block
             | observables_block'''
    p[0] = p[1]

def p_block_empty(p):
    '''block : NEWLINE'''

def p_parameter_block(p):
    'parameter_block : BEGIN PARAMETERS NEWLINE parameter_st_list END PARAMETERS NEWLINE'
    p[0] = p[4]
    print("block:", p[2])

def p_molecule_type_block(p):
    'molecule_type_block : BEGIN MOLECULE_TYPES NEWLINE END MOLECULE_TYPES NEWLINE'
    p[0] = p[2]
    print("block:", p[2])

def p_species_block(p):
    'species_block : BEGIN SPECIES NEWLINE END SPECIES NEWLINE'
    p[0] = p[2]
    print("block:", p[2])

def p_reaction_rules_block(p):
    'reaction_rules_block : BEGIN REACTION_RULES NEWLINE END REACTION_RULES NEWLINE'
    p[0] = p[2]
    print("block:", p[2])

def p_observables_block(p):
    'observables_block : BEGIN OBSERVABLES NEWLINE END OBSERVABLES NEWLINE'
    p[0] = p[2]
    print("block:", p[2])

def p_parameter_st_list(p):
    '''parameter_st_list : parameter_st_list parameter_st
                         | parameter_st
                         | '''
    list_helper(p)

def p_parameter_st(p):
    '''parameter_st : INTEGER ID number NEWLINE
                    | NEWLINE'''
    if len(p) > 2:
        p[0] = p[1:4]

def p_number(p):
    '''number : FLOAT
              | INTEGER'''
    p[0] = p[1]

# def p_statement_list(p):
#     '''statement_list : statement_list statement'''
#     p[0] = p[1] + [p[2]]
#     #print("statement_list:", p[0])

# def p_statement_list_trivial(p):
#     '''statement_list : statement'''
#     p[0] = [p[1]]
#     #print("statement_list_trivial:", p[0])

# def p_statement_empty(p):
#     'statement : NEWLINE'
#     #print("statement_empty:", p[0])

# def p_statement(p):
#     'statement : rule NEWLINE'
#     p[0] = p[1]
#     #print("statement:", p[0])

# def p_rule(p):
#     '''rule : irr_rule
#             | rev_rule'''
#     p[0] = p[1]
#     #print("rule:", p[0])

# def p_irr_rule(p):
#     'irr_rule : expression IRRARROW expression LPAREN FLOAT RPAREN'
#     #print("irr_rule")
#     p[0] = RuleIrreversible(reactants=p[1], products=p[3], rate=p[5])

# def p_rev_rule(p):
#     'rev_rule : expression REVARROW expression LPAREN FLOAT COMMA FLOAT RPAREN'
#     #print("rev_rule")
#     p[0] = RuleReversible(reactants=p[1], products=p[3], rates=[p[5], p[7]])

# def p_expression_plus(p):
#     'expression : expression PLUS expression'
#     p[0] = p[1] + p[3]
#     #print("expression_plus:", p[0])

# def p_expression_species(p):
#     'expression : SPECIES'
#     #print("expression_species:", p[1])
#     p[0] = [Species(name=p[1])]

# Error rule for syntax errors
def p_error(p):
    print("Syntax error in input:")
    print(p)

precedence = (
    ('left', 'PLUS'),
)

# Build the parser
lex.lex()
yacc.yacc(write_tables=0)


def parse(*args, **kwargs):
    yacc.parse(*args, **kwargs)
