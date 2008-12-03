from ply import yacc, lex
import toyyacc

rules = '''
E + S <-> ES (1,1)
ES --> E + P (1)
'''

model = yacc.parse(rules)

print
print "rules:"
for r in model.rules:
    print ' ', r
print
