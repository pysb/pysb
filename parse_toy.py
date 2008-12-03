from ply import yacc, lex


rules = '''
A + B --> C (0.5) # blah #

#this is all a comment
B + D <-> C (1.5, .9) #comment 
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
