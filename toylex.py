from ply import lex;

tokens = (
    'FLOAT',
    'SPECIES',
    'PLUS',
    'IRRARROW',
    'REVARROW',
    'LPAREN',
    'RPAREN',
    'COMMA',
    'NEWLINE',
    )

t_SPECIES  = r'\w+'
t_PLUS     = r'\+'
t_IRRARROW = r'-->'
t_REVARROW = r'<->'
t_LPAREN   = r'\('
t_RPAREN   = r'\)'
t_COMMA    = r','

# Match and ignore comments (# to end of line)
def t_comment(t):
    r'\#[^\n]*'

# Define a rule so we can track line numbers
def t_NEWLINE(t):
    r'\n'
    t.lexer.lineno += 1
    return t

def t_FLOAT(t):
    r'(\d+(\.\d+)?|\.\d+)'
    try:
        t.value = float(t.value)    
    except ValueError:
        print "Line %d: Number '%s' has some kind of problem (ValueError)!" % (t.lineno,t.value)
        t.value = float("nan")
    return t

# A string containing ignored characters (spaces and tabs)
t_ignore  = ' \t'

# Error handling rule
def t_error(t):
    print "Illegal character '%s' on line %d" % (t.value[0], t.lineno)
    t.lexer.skip(1)

lex.lex()
