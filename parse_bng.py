import parser.bng as bng_parser

model = '''
begin parameters
 1 R0   1
 2 kp1  0.5
 3 km1  0.1
 4 kp2  1e-3
 5 km2  0.1
 6 p1  10
 7 d1   5
 8 kpA  1e-4
 9 kmA  0.02
end parameters

begin species
end species

begin reaction_rules
end reaction_rules

begin observables
end observables
'''

bng_parser.parse(model)

#t = bng_parser.lex.input(model)
#while 1:
#    tok = bng_parser.lex.token()
#    if not tok: break      # No more input
#    print tok
