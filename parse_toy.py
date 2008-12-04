from ply import yacc, lex
import toyyacc

rules = '''
E + S <-> ES (1,2)
ES --> E + P (0.3)
'''

model = yacc.parse(rules)

print
print "rules:"
for r in model.rules:
    print ' ', r

model.finalize()

print
print "equations:"
for s in model.species.values():
    print '  d[%s]/dt = %s' % (s.name, ' + '.join(str(t) for t in s.mass_action_terms))
