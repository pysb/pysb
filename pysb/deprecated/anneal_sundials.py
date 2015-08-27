from __future__ import print_function
import pysb.bng
import numpy 
import sympy 
import re 
import ctypes
import csv
import scipy.interpolate
from pysundials import cvode

# These set of functions set up the system for annealing runs
# and provide the runner function as input to annealing

def annlinit(model, abstol=1.0e-3, reltol=1.0e-3, nsteps = 1000, itermaxstep = None):
    '''
    must be run to set up the environment for annealing with pysundials
    '''
    # Generate equations
    pysb.bng.generate_equations(model)
    # Get the size of the ODE array
    odesize = len(model.odes)
    
    # init the arrays we need
    yzero = numpy.zeros(odesize)  #initial values for yzero
    
    # assign the initial conditions
    for cplxptrn, ic_param in model.initial_conditions:
        speci = model.get_species_index(cplxptrn)
        yzero[speci] = ic_param.value

    # initialize y with the yzero values
    y = cvode.NVector(yzero)
        
    # make a dict of ydot functions. notice the functions are in this namespace.
    # replace the kxxxx constants with elements from the params array
    rhs_exprs = []
    for i in range(0,odesize):
        # first get the function string from sympy, replace the the "sN" with y[N]
        tempstring = re.sub(r's(\d+)', lambda m: 'y[%s]'%(int(m.group(1))), str(model.odes[i]))
        # now replace the constants with 'p' array names; cycle through the whole list
        #for j in range(0, numparams):
        #    tempstring = re.sub('(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])'%(model.parameters[j].name),'p[%d]'%(j), tempstring)
        for j, parameter in enumerate(model.parameters):
            tempstring = re.sub('(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])' % parameter.name, 'p[%d]' % j, tempstring)
        # make a list of compiled rhs expressions which will be run by the integrator
        # use the ydots to build the function for analysis
        # (second arg is the "filename", useful for exception/debug output)
        rhs_exprs.append(compile(tempstring, '<ydot[%s]>' % i, 'eval'))
    
    # Create the structure to hold the parameters when calling the function
    # This results in a generic "p" array
    numparams = len(model.parameters)
    class UserData(ctypes.Structure):
        _fields_ = [('p', cvode.realtype*numparams)] # parameters
    PUserData = ctypes.POINTER(UserData)
    data = UserData() 

    #paramlist for annealing feeder function
    #paramlist = []
    #for i in range(0, numparams):
    #    # notice: p[i] ~ model.parameters[i].name ~ model.parameters[i].value
    #    data.p[i] = model.parameters[i].value
    #    paramlist.append(model.parameters[i].value)
    #paramarray = numpy.asarray(paramlist)

    data.p[:] = [p.value for p in model.parameters]
    paramarray = numpy.array([p.value for p in model.parameters])
    
    # if no sensitivity analysis is needed allocate the "p" array as a 
    # pointer array that can be called by sundials "f" as needed
    def f(t, y, ydot, f_data):
        data = ctypes.cast(f_data, PUserData).contents
        rhs_locals = {'y': y, 'p': data.p}
        for i in range(0,len(model.odes)):
            ydot[i] = eval(rhs_exprs[i], rhs_locals)
        return 0
    
    # CVODE STUFF
    # initialize the cvode memory object, use BDF and Newton for stiff
    cvode_mem = cvode.CVodeCreate(cvode.CV_BDF, cvode.CV_NEWTON)
    # allocate the cvode memory as needed, pass the function and the init ys
    cvode.CVodeMalloc(cvode_mem, f, 0.0, y, cvode.CV_SS, reltol, abstol)
    # point the parameters to the correct array
    # if the params are changed later this does not need to be reassigned (???)
    cvode.CVodeSetFdata(cvode_mem, ctypes.pointer(data))
    # link integrator with linear solver
    cvode.CVDense(cvode_mem, odesize)
    #stepsize
    if itermaxstep != None:
        cvode.CVodeSetMaxStep(cvode_mem, itermaxstep)

    #list of outputs
    xout = numpy.zeros(nsteps)
    yout = numpy.zeros([nsteps, odesize])

    #initialize the arrays
    #print("Initial parameter values:", y)
    xout[0] = 0.0 #CHANGE IF NEEDED
    #first step in yout
    for i in range(0, odesize):
        yout[0][i] = y[i]

    # f: the function called by cvodes calls that returns dy
    # rhs_exprs: the python expression for the right hand side (list of strings)
    # y: a CVode NVector object with the initial values for all species
    # odesize: the number of odes
    # data: a ctypes data structure (for Sundials) containing the parameter values (floats)
    # xout: the numpy array where the time values will be put
    # yout: the numpy array where the integrated timeseries will be put
    # nsteps: the number of time steps
    # cvode_mem: cvode memory object defining the step method
    # yzero: a numpy array of the initial conditions
    # paramarray: a numpy array containing the parameter values (floats). (same contents as data, but different datatype)
    # reltol: integrator relative tolerance (float)
    # abstol: integrator absolute tolerance (float)
    return [f, rhs_exprs, y, odesize, data, xout, yout, nsteps, cvode_mem, yzero, reltol, abstol], paramarray


# reltol of 1.0e-3, relative error of ~1%. abstol of 1.0e-3, enough for values that oscillate in the hundreds to thousands
def annlodesolve(model, tfinal, envlist, params, tinit = 0.0, ic=True):
    '''
    the ODE equation solver tailored to work with the annealing algorithm
    model: the model object
    envlist: the list returned from annlinit
    params: the list of parameters that are being optimized with annealing 
    tinit: initial time
    reltol: relative tolerance
    abstol: absolute tolerance
    ic: reinitialize initial conditions to a value in params 
    '''
    (f, rhs_exprs, y, odesize, data, xout, yout, nsteps, cvode_mem, yzero, reltol, abstol) = envlist

    #set the initial values and params in each run
    #all parameters are used in annealing. 
    for i in range(len(params)):
        data.p[i] = params[i]
        
    # update yzero if initial conditions are being modified as part of the parameters
    # did it this way b/c yzero and data.p may not always want to be modified at the same time
    # FIXME: this is not the best way to do this.
    # the params list should NOT contain the initial conditions if they are not
    # to be used in the annealing... so this is a hack based on the fact that the
    # initial conditions are contained as part of the model.parameters list.
    # FIXME
    #
    if ic is True:
        for cplxptrn, ic_param in model.initial_conditions:
            speci = model.get_species_index(cplxptrn)
            yzero[speci] = ic_param.value
            
    #reset initial concentrations
    y = cvode.NVector(yzero)

    # Reinitialize the memory allocations, DOES NOT REALLOCATE
    cvode.CVodeReInit(cvode_mem, f, 0.0, y, cvode.CV_SS, reltol, abstol)
    
    tadd = tfinal/nsteps

    t = cvode.realtype(tinit)
    tout = tinit + tadd
    
    #print("Beginning integration")
    #print("TINIT:", tinit, "TFINAL:", tfinal, "TADD:", tadd, "ODESIZE:", odesize)
    #print("Integrating Parameters:\n", params)
    #print("y0:", yzero)

    for step in range(1, nsteps):
        ret = cvode.CVode(cvode_mem, tout, y, ctypes.byref(t), cvode.CV_NORMAL)
        if ret !=0:
            print("CVODE ERROR %i"%(ret))
            break

        xout[step]= tout
        for i in range(0, odesize):
            yout[step][i] = y[i]

        # increase the time counter
        tout += tadd
    #print("Integration finished")

    #now deal with observables
    yobs = numpy.zeros([len(model.observables), nsteps])
    
    #sum up the correct entities
    for i, obs in enumerate(model.observables):
        coeffs = obs.coefficients
        specs  = obs.species
        yobs[i] = (yout[:, specs] * coeffs).sum(1)

    #merge the x and y arrays for easy analysis
    xyobs = numpy.vstack((xout, yobs))

    return (xyobs,xout,yout, yobs)

def compare_data(xparray, simarray, xspairlist, vardata=False):
    """Compares two arrays of different size and returns the X^2 between them.
    Uses the X axis as the unit to re-grid both arrays. 
    xparray: experimental data
    xparrayaxis: which axis of xparray to use for simulation
    simarray: simulation data
    simarrayaxis: which axis of simarray to use for simulation
    """
    # this expects arrays of the form array([time, measurement1, measurement2, ...])
    # the time is assumed to be roughly the same for both and the 
    # shortest time will be taken as reference to regrid the data
    # the regridding is done using a b-spline interpolation
    # xparrayvar shuold be the variances at every time point
    #
    # FIXME FIXME FIXME FIXME
    # This prob should figure out the overlap of the two arrays and 
    # get a spline of the overlap. For now just assume the simarray domain
    # is bigger than the xparray. FIXME FIXME FIXME FIXME 
    #
    #rngmin = max(xparray[0].min(), simarray[0].min())
    #rngmax = min(xparray[0].max(), simarray[0].max())
    #rngmin = round(rngmin, -1)
    #rngmax = round(rngmax, -1)
    #print("Time overlap range:", rngmin,"to", rngmax)
    
    ipsimarray = numpy.zeros(xparray.shape[1])
    objout = 0
   
    for i in range(len(xspairlist)):
        # create a b-spline of the sim data and fit it to desired range
        
        #some error checking
        #print("xspairlist length:", len(xspairlist[i]))
        #print("xspairlist element type:", type(xspairlist[i]))
        #print("xspairlist[i] elements:", xspairlist[i][0], xspairlist[i][1])
        assert type(xspairlist[i]) is tuple
        assert len(xspairlist[i]) == 2
        
        xparrayaxis = xspairlist[i][0]
        simarrayaxis = xspairlist[i][1]
        
        tck = scipy.interpolate.splrep(simarray[0], simarray[simarrayaxis])
        ipsimarray = scipy.interpolate.splev(xparray[0], tck) #xp x-coordinate values to extract from y splines
        
        # we now have x and y axis for the points in the model array
        # calculate the objective function
        #                        1
        # obj(t, params) = -------------(S_sim(t,params)-S_exp(t))^2
        #                  2*sigma_exp^2
        
        diffarray = ipsimarray - xparray[xparrayaxis]
        diffsqarray = diffarray * diffarray

        if vardata is True:
            #print("using XP VAR",xparrayaxis+1)
            xparrayvar = xparray[xparrayaxis+1] # variance data provided in xparray in next column
        else:
        # assume a default variance
            xparrayvar = numpy.ones(xparray.shape[1])
            xparrayvar = xparray[xparrayaxis]*.25 #assume a coeff of variation of .25 = sigma/mean (from Chen, Gaudet...)
            # Remove any zeros in the variance array # FIXME
            for i in range(0, len(xparrayvar)):
                if (xparrayvar[i] == 0):
                    xparrayvar[i] = 1

        xparrayvar = xparrayvar*2.0
        #numpy.seterr(divide='ignore')
        objarray = diffsqarray / xparrayvar

        # check for inf in objarray, they creep up when there are near zero or zero values in xparrayvar
        for i in range(len(objarray)):
            if numpy.isinf(objarray[i]) or numpy.isnan(objarray[i]):
                #print("CORRECTING NAN OR INF. IN ARRAY")
                #print(objarray)
                objarray[i] = 1e-100 #zero enough

        objout += objarray.sum()
        #print("OBJOUT(%d,%d):%f  |\t\tOBJOUT(CUM):%f"%(xparrayaxis, simarrayaxis, objarray.sum(), objout))

    print("OBJOUT(total):", objout)
    return objout

def logparambounds(params, omag=1, useparams=[], usemag=None, initparams=[], initmag=None):
    ub = numpy.zeros(len(params))
    lb = numpy.zeros(len(params))
    # set upper/lower bounds for generic problem
    for i in range(len(params)):
        if i in useparams and i not in initparams:
            ub[i] = params[i] * pow(10,usemag)
            lb[i] = params[i] / pow(10,usemag)
        elif i in initparams:
            ub[i] = params[i] * pow(10,initmag)
            lb[i] = params[i] / pow(10,initmag)
        else:
            ub[i] = params[i] * pow(10, omag)
            lb[i] = params[i] / pow(10, omag)
    return lb, ub

def linparambounds(params, fact=.25, useparams=[], usefact=None):
    ub = numpy.zeros(len(params))
    lb = numpy.zeros(len(params))
    # set upper/lower bounds for generic problem
    for i in range(len(params)):
        if i in useparams:
            ub[i] = params[i] + (params[i] * fact)
            lb[i] = params[i] - (params[i] * fact)
        else:
            ub[i] = params[i] + (params[i] * usefact)
            lb[i] = params[i] - (params[i] * usefact)
    lb[numpy.where(lower<0.)] = 0.0 #make sure we don't go negative on parameters...
    return lb, ub

def mapprms(nums01, lb, ub, scaletype="log"):
    """given an upper bound(ub), lower bound(lb), and a sample between zero and one (zosample)
    return a set of parameters within the lb, ub range. 
    nums01: array of numbers between zero and 1
    lb: array of lower bound for each parameter
    ub: arary of upper bound for each parameter
    """
    params = numpy.zeros_like(nums01)
    
    if scaletype == "log":
        params = lb*(ub/lb)**nums01 # map the [0..1] array to values sampled over their omags
    elif scaletype == "lin":
        params = (nums01*(ub-lb)) + lb

    return params

def writetofile(fout, simparms, simdata, temperature):
    imax, jmax = simdata.shape
    nparms = len(simparms)

    fout.write('# TEMPERATURE\n{0}\n'.format(temperature))
    fout.write('# PARAMETERS ({0})\n'.format(len(simparms)))
    for i in range(nparms):
        fout.write('{0}'.format(simparms[i]))
        if (i !=0 and i%5 == 0) or (i == nparms-1):
            fout.write('\n')
        else:
            fout.write(', ')
            
    fout.write('# SIMDATA ({0},{1})\n'.format(imax, jmax))
    for i in range(imax):
        fout.write('# {0}\n'.format(i))
        for j in range(jmax):
            fout.write('{0}'.format(simdata[i][j]))
            if (j != 0 and j%10 == 0) or (j == jmax-1):
                fout.write('\n')
            else:
                fout.write(', ')
    fout.write('#-------------------------------------------------------------------------------------------------\n')
    return


def tenninetycomp(outlistnorm, arglist, xpsamples=1.0):
    """ Determine Td and Ts. Td calculated at time when signal goes up to 10%.
        Ts calculated as signal(90%) - signal(10%). Then a chi-square is calculated.
        outlistnorm: the outlist from anneal_odesolve
        arglist: simaxis, Tdxp, varTdxp, Tsxp, varTsxp
        xpsamples
    """
    xarr = outlistnorm[0] #this assumes the first column of the array is time
    yarr = outlistnorm[arglist[0]] #the argument passed should be the axis
    Tdxp = arglist[1]
    varTdxp = arglist[2]
    Tsxp = arglist[3]
    varTsxp = arglist[4]
    
    # make a B-spine representation of the xarr and yarr
    tck = scipy.interpolate.splrep(xarr, yarr)
    t, c, k = tck
    tenpt = numpy.max(yarr) * .1 # the ten percent point in y-axis
    ntypt = numpy.max(yarr) * .9 # the 90 percent point in y-axis
    #lower the spline at the abcissa
    xten = scipy.interpolate.sproot((t, c-tenpt, k))[0]
    xnty = scipy.interpolate.sproot((t, c-ntypt, k))[0]

    #now compare w the input data, Td, and Ts
    Tdsim = xten #the Td is the point where the signal crosses 10%; should be the midpoint???
    Tssim = xnty - xten
    
    # calculate chi-sq as
    # 
    #            1                           1
    # obj = ----------(Tdsim - Tdxp)^2 + --------(Tssim - Tsxp)^2
    #       2*var_Tdxp                   2*var_Td 
    #
    
    obj = ((1./varTdxp) * (Tdsim - Tdxp)**2.) + ((1./varTsxp) * (Tssim - Tsxp)**2.)
    #obj *= xpsamples
    
    print("OBJOUT-10-90:(%g,%g):%g"%(Tdsim, Tssim, obj))

    return obj    

def annealfxn(zoparams, time, model, envlist, xpdata, xspairlist, lb, ub, tn = [], scaletype="log", norm=True, vardata=False, fileobj=None):
    """Feeder function for scipy.optimize.anneal
    zoparams: the parameters in the range [0,1) to be sampled
    time: the time scale for the simulation
    model: a PySB model object
    envlist: an environment list for the sundials integrator
    xpdata: experimental data
    xspairlist: the pairlist of the correspondence of experimental and simulation outputs
    lb: lower bound for parameters
    ub: upper bound for parameters
    tn: list of values for ten-ninety fits
    scaletype: log, linear,etc to convert zoparams to real params b/w lb and ub. default "log"
    norm: normalization on. default true
    vardata: variance data available. default "false"
    fileobj: file object to write data output. default "None"

    """

    # convert of linear values from [0,1) to desired sampling distrib
    paramarr = mapprms(zoparams, lb, ub, scaletype="log")

    #debug
    #print("ZOPARAMS:\n", zoparams,"\n")
    #print(paramarr)

    # eliminate values outside the boundaries, i.e. those outside [0,1)
    if numpy.greater_equal(paramarr, lb).all() and numpy.less_equal(paramarr, ub).all():
        print("integrating... ")
        outlist = annlodesolve(model, time, envlist, paramarr)

        # normalized data needs a bit more tweaking before objfxn calculation
        if norm is True:
            print("Normalizing data")
            datamax = numpy.max(outlist[0], axis = 1)
            datamin = numpy.min(outlist[0], axis = 1)
            outlistnorm = ((outlist[0].T - datamin)/(datamax-datamin)).T
            # xpdata[0] should be time, get from original array
            outlistnorm[0] = outlist[0][0].copy()
            # xpdata here should be normalized
            objout = compare_data(xpdata, outlistnorm, xspairlist, vardata)
            if tn:
                tn = tenninetycomp(outlistnorm, tn, len(xpdata[0]))
                objout += tn 
            print("NORM objout TOT:", objout)
        else:
            objout = compare_data(xpdata, outlist[0], xspairlist, vardata)
            if tn:
                tn = tenninetycomp(outlist[0], tn)
                objout += tn 
            print("objout TOT:", objout)
    else:
        print("======> VALUE OUT OF BOUNDS NOTED")
        temp = numpy.where((numpy.logical_and(numpy.greater_equal(paramarr, lb), numpy.less_equal(paramarr, ub)) * 1) == 0)
        for i in temp:
            print("======>",i,"\n======", paramarr[i],"\n", zoparams[i],"\n")
        objout = 1.0e300 # the largest FP in python is 1.0e308, otherwise it is just Inf

    # save the params and temps for analysis

    # FIXME If a parameter is out of bounds, outlist and outlistnorm will be undefined and this will cause an error
    if fileobj:
        if norm:
            writetofile(fileobj, paramarr, outlistnorm, objout)
        else:
            writetofile(fileobj, paramarr, outlist, objout)
    
    return objout


