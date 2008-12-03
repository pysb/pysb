from ply import yacc, lex


rules = '''
E + S <-> ES (1,1)
ES --> E + P (1)
'''

#import toylex
#lex.input(rules)
#while 1:
#    tok = lex.token()
#    if not tok: break      # No more input
#    print tok


import toyyacc
model = yacc.parse(rules)

print "\nrules:"
for r in model.rules:
    print r
