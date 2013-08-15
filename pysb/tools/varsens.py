
import pysb.bng
import ghalton
import numpy
import sys
from sobol_seq import i4_sobol_generate

def varsens(objective, k, n, scaling, log_scaling=False, verbose=True):
    if verbose: print "Generating Low Discrepancy Sequence"
    
    # This is a known working strategy
    #seq = ghalton.Halton(k) # half for A, and half for B
    #seq.get(2*(k*k-k)) # Burn away any face exploration off the Halton
    #M_1  = scale(numpy.array(seq.get(n)), scaling, log_scaling)  # See Eq (9)
    #x = numpy.transpose(i4_sobol_generate(k, n, k+numpy.random.randint(2**14)))
    #M_2  = scale(x, scaling, log_scaling)

    # This appears to work!
    seq = ghalton.Halton(k*2)
    seq.get(20*k) # Remove initial linear correlation
    x = numpy.array(seq.get(n))
    M_1 = scale(x[...,0:k    ], scaling, log_scaling) 
    M_2 = scale(x[...,k:(2*k)], scaling, log_scaling) 
    
    N_j  = generate_N_j(M_1, M_2)                                # See Eq (11)
    N_nj = generate_N_j(M_2, M_1)
    
    (fM_1, fM_2, fN_j, fN_nj) = objective_values(M_1, M_2, N_j, N_nj, objective, verbose) 
    
    if verbose: print "Final sensitivity calculation"
    return getvarsens(fM_1, fM_2, fN_j, fN_nj)

def scale(points, scaling, log_scaling):
    if log_scaling:
# FIXME, I THINK THIS IS ALL BACKWARD, Ugh.
        s = numpy.exp(scaling)
        return numpy.log(points*(s[1]-s[0]) + s[0])
    else:
        return points * (scaling[1] - scaling[0]) + scaling[0]

def move_spinner(i):
    spin = ("|", "/","-", "\\")
    print "[%s] %d\r"%(spin[i%4],i),
    sys.stdout.flush()

def generate_N_j(M_1, M_2):
    """when passing the quasi-random low discrepancy-treated A and B matrixes, this function
    iterates over all the possibilities and returns the C matrix for simulations.
    See e.g. Saltelli, Ratto, Andres, Campolongo, Cariboni, Gatelli, Saisana,
    Tarantola Global Sensitivity Analysis"""

    nparams = M_1.shape[1] # shape 1 should be the number of params

    # allocate the space for the C matrix
    N_j = numpy.array([M_2]*nparams) 

    # Now we have nparams copies of M_2. replace the i_th column of N_j with the i_th column of M_1
    for i in range(nparams):
        N_j[i,:,i] = M_1[:,i]

    return N_j

def objective_values(M_1, M_2, N_j, N_nj, objective, verbose=True): #, fileobj=None):
    ''' Function parmeval calculates the fM_1, fM_2, and fN_j_i arrays needed for variance-based
    global sensitivity analysis as prescribed by Saltelli and derived from the work by Sobol
    (low-discrepancy sequences)
    '''

    # assign the arrays that will hold fM_1, fM_2 and fN_j_n
    fM_1  = numpy.zeros(M_1.shape[0])
    fM_2  = numpy.zeros(M_2.shape[0])
    fN_j  = numpy.zeros([M_1.shape[1]] + [M_1.shape[0]]) # matrix is of shape (nparam, nsamples)
    fN_nj = numpy.zeros([M_1.shape[1]] + [M_1.shape[0]])

    # First process the A and B matrices
    if verbose: print "Processing f(M_1):"
    for i in range(M_1.shape[0]):
        fM_1[i]   = objective(M_1[i])
        if verbose: move_spinner(i)

    if verbose: print "Processing f(M_2):"
    for i in range(M_2.shape[0]):
        fM_2[i]   = objective(M_2[i])
        if verbose: move_spinner(i)

    if verbose: print "Processing f(N_j)"
    for i in range(N_j.shape[0]):
        if verbose: print " * parameter %d"%i
        for j in range(N_j.shape[1]):
            fN_j[i][j] = objective(N_j[i][j])
            if verbose: move_spinner(j)

    if verbose: print "Processing f(N_nj)"
    for i in range(N_j.shape[0]):
        if verbose: print " * parameter %d"%i
        for j in range(N_j.shape[1]):
            fN_nj[i][j] = objective(N_nj[i][j])
            if verbose: move_spinner(j)

    return fM_1, fM_2, fN_j, fN_nj

def getvarsens(fM_1, fM_2, fN_j, fN_nj):
    nparms   = fN_j.shape[0] # should be the number of parameters
    nsamples = fN_j.shape[1] # should be the number of samples from the original matrix

    E_2 = sum(fM_1*fM_2) / nsamples      # Eq (21)

    # Estimate U_j and U_-j values and store them 
    U_j  = numpy.sum(fM_1 * fN_j,  axis=1) / (nsamples - 1)  # Eq (12)
    U_nj = numpy.sum(fM_1 * fN_nj, axis=1) / (nsamples - 1)  # Eq (unnumbered one after 18)

    #estimate V(y) from fM_1 and fM_2, paper uses only fM_1, this is a better estimate
    var_y = (numpy.var(fM_1, axis=0, ddof=1)+numpy.var(fM_2, axis=0, ddof=1))/2.0

    #allocate the S_i and ST_i arrays
    Sens  = numpy.zeros(nparms)
    SensT = numpy.zeros(nparms)

    # now get the S_i and ST_i, Eq (27) & Eq (28)
    for j in range(nparms):
        Sens[j]  =       ((U_j[j] - E_2) / var_y)
        SensT[j] = 1.0 - ((U_nj[j]- E_2) / var_y)

    return Sens, SensT, var_y, E_2


# Working on a test function here

# This is defined on the range [0..1]
# Eq (29)
def g_function(x, a):
    return numpy.prod([gi_function(xi, a[i]) for i,xi in enumerate(x)])

# Eq (30), Validated
def gi_function(xi, ai):
    return (numpy.abs(4.0*xi-2.0)+ai) / (1.0+ai)

model = [0, 0.5, 3, 9, 99, 99]

# Analytical answer, Eq (34) divided by V(y), matches figure
answer = 1.0/(3.0* ((numpy.array(model) + 1.0)**2.0))
numpy.round(answer, 3)


def g_objective(x): return g_function(x, model)

v = varsens(g_objective, 6, 1024, numpy.array([[0.0]*6, [1.0]*6]))

# http://www.jstor.org/stable/pdfplus/2676831.pdf
#from numpy.polynomial.legendre import legval