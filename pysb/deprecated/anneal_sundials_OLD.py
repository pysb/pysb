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

def annlinit(model, abstol=1.0e-3, reltol=1.0e-3, nsteps = 20000, itermaxstep = None):
    """
    annlinit initializes the environment for simulations using CVODE
    INPUT:
    -------
    model: a PySB model object
    abstol: absolute tolerance for CVODE integrator
    reltol: relative tolerance for CVODE integrator
    nsteps: number of time steps of integration (depends on the units of the parameters)
    itermaxstep: maximum number of iteration steps

    OUTPUT:
    -------
    a list object containing:
    [f, rhs_exprs, y, odesize, data, xout, yout, nsteps, cvode_mem, yzero, reltol, abstol]
       f: the function called by cvodes calls that returns dy
       rhs_exprs: the python expression for the right hand side (list of strings)
       y: a CVode NVector object with the initial values for all species
       odesize: the number of odes
       data: a ctypes data structure (for Sundials) containing the parameter values (floats)
       xout: the numpy array where the time values will be put
       yout: the numpy array where the integrated timeseries will be put
       nsteps: the number of time steps
       cvode_mem: cvode memory object defining the step method
       yzero: a numpy array of the initial conditions
       paramarray: a numpy array containing the parameter values (floats). (same contents as data, but different datatype)
       reltol: integrator relative tolerance (float)
       abstol: integrator absolute tolerance (float)

    paramarray, a parameter array
    """

    # Generate equations
    pysb.bng.generate_equations(model)
    odesize = len(model.odes)
    
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
        # get the function string from sympy, replace the the "sN" with y[N]
        tempstring = re.sub(r's(\d+)', lambda m: 'y[%s]'%(int(m.group(1))), str(model.odes[i]))

        # replace the constants with 'p' array names; cycle through the whole list
        for j, parameter in enumerate(model.parameters):
            tempstring = re.sub('(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])' % parameter.name, 'p[%d]' % j, tempstring)

        # make a list of compiled rhs expressions which will be run by the integrator
        # use the ydots to build the function for analysis
        # (second arg is the "filename", useful for exception/debug output)
        rhs_exprs.append(compile(tempstring, '<ydot[%s]>' % i, 'eval'))
    
    # Create a generic "p" array to hold the parameters when calling the function
    numparams = len(model.parameters)
    class UserData(ctypes.Structure):
        _fields_ = [('p', cvode.realtype*numparams)] # parameters
    PUserData = ctypes.POINTER(UserData)
    data = UserData() 

    # Create the paramarray to hold the parameters and pass them from C to Python
    # FIXME: could prob do this directly from data.p
    data.p[:] = [p.value for p in model.parameters]
    paramarray = numpy.array([p.value for p in model.parameters])
    
    # allocate "p" as a pointer array that can be called by sundials "f" as needed
    def f(t, y, ydot, f_data):
        data = ctypes.cast(f_data, PUserData).contents
        rhs_locals = {'y': y, 'p': data.p}
        for i in range(0,len(model.odes)):
            ydot[i] = eval(rhs_exprs[i], rhs_locals)
        return 0
    
    # initialize the cvode memory object, use BDF and Newton for stiff
    cvode_mem = cvode.CVodeCreate(cvode.CV_BDF, cvode.CV_NEWTON)
    # allocate the cvode memory as needed, pass the function and the initial ys
    cvode.CVodeMalloc(cvode_mem, f, 0.0, y, cvode.CV_SS, reltol, abstol)
    # point the parameters to the correct array
    cvode.CVodeSetFdata(cvode_mem, ctypes.pointer(data))
    # link integrator with linear solver
    cvode.CVDense(cvode_mem, odesize)
    # maximum iteration steps
    if itermaxstep != None:
        cvode.CVodeSetMaxStep(cvode_mem, itermaxstep)

    #list of outputs
    xout = numpy.zeros(nsteps)
    yout = numpy.zeros([nsteps, odesize])

    #initialize the arrays
    xout[0] = 0.0 # FIXME: this assumes that the integration starts at zero... CHANGE IF NEEDED
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


# reltol of 1.0e-3, relative error of ~1%. abstol of 1.0e-2, enough for values that oscillate in the hundreds to thousands
def annlodesolve(model, tfinal, envlist, params, useparams=None, tinit = 0.0, ic=True):
    '''
    the ODE equation solver tailored to work with the annealing algorithm
    model: the model object
    envlist: the list returned from annlinit
    params: the list of parameters that are being optimized with annealing 
    useparams: the parameter number to which params[i] corresponds
    tinit: initial time
    reltol: relative tolerance
    abstol: absolute tolerance
    ic: reinitialize initial conditions to a value in params or useparams
    '''
    (f, rhs_exprs, y, odesize, data, xout, yout, nsteps, cvode_mem, yzero, reltol, abstol) = envlist

    #set the initial values and params in each run
    #all parameters are used in annealing. initial conditions are not, here
    if useparams is None:
        for i in range(len(params)):
            data.p[i] = params[i]
    else:
        #only a subset of parameters are used for annealing
        for i in range(len(useparams)):
            #print("changing parameter", model.parameters[useparams[i]],"data.p", data.p[useparams[i]],"to", params[i])
            data.p[useparams[i]] = params[i]

    # update yzero if initial conditions are being modified as part of the parameters
    # did it this way b/c yzero and data.p may not always want to be modified at the same time
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
    obs_names = [name for name, rp in model.observable_patterns]
    yobs = numpy.zeros([len(obs_names), nsteps])
    
    #sum up the correct entities
    for i, name in enumerate(obs_names):
        factors, species = zip(*model.observable_groups[name])
        yobs[i] = (yout[:, species] * factors).sum(axis = 1)

    #merge the x and y arrays for easy visualization
    xyobs = numpy.vstack((xout, yobs))

    return (xyobs,xout,yout, yobs)

def compare_data(xparray, simarray, xspairlist, vardata=False):
    """Compares two arrays of different size and returns the X^2 between them.
    Uses the X axis as the unit to re-grid both arrays. 
    xparray: experimental data
    simarray: simulation data
    xspairlist: list of pairs of xp data and sim data that go together
    vardata: TRUE if xparray contains variance data
    
    """
    # expects arrays of the form array([time, measurement1, measurement2, ...])
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
        # import code
        # code.interact(local=locals())
        
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
            xparrayvar = xparray[xparrayaxis]*.341 # 1 stdev w/in 1 sigma of the experimental data... 
            xparrayvar = xparrayvar * xparrayvar
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

        #import code
        #code.interact(local=locals())

        objout += objarray.sum()
        print("OBJOUT(%d,%d):%f  |\t\tOBJOUT(CUM):%f"%(xparrayaxis, simarrayaxis, objarray.sum(), objout))

    print("OBJOUT(total):", objout)
    return objout

def getgenparambounds(params, omag=1, N=1000., useparams=[], usemag=None, useN=None ):
    # params must be a numpy array
    # from: http://projects.scipy.org/scipy/ticket/1126
    # The input-parameters "lower" and "upper" do not refer to global bounds of the
    # parameter space but to 'maximum' displacements in the MC scheme AND
    # in addition they determine the initial point!! 
    # The initial value that you provide for the parameter vector seems to have no influence.
    # This is how I call anneal with my desired functionality
    # p=[a,b,c] #my initial values
    # lb=array([a0,b0,c0]) #my lower bounds (absolute values)
    # ub=array([a1,b1,c1]) #my upper bounds
    # N=100 #determines the size of displacements; the more N, the smaller steps and the longer time to convergence
    # dx=(ub-lb)/N #displacements--you could get from ub to lb in N steps
    # lower=array(p)-dx/2 #the "lower bound" of the anneal routine (upper and lower step bounds relative to the current values of p)
    # upper=array(p)+dx/2 #the "upper bound" of the anneal routine
    # f=lambda var: costfunction(p,lb,ub) #my cost function that is made very high if not lb < p < ub
    # pbest=scipy.optimize.anneal(f,p,lower=lower,upper=upper)
    # This ensures a MC search that starts of close to my initial value and makes steps of dx in its search.
    ub = numpy.zeros(len(params))
    lb = numpy.zeros(len(params))
    dx = numpy.zeros(len(params))
    # set upper/lower bounds for generic problem
    for i in range(len(params)):
        if i in useparams:
            ub[i] = params[i] * pow(10,usemag)
            lb[i] = params[i] / pow(10,usemag)
            dx[i] = (ub[i] - lb[i])/useN
        else:
            ub[i] = params[i] * pow(10, omag)
            lb[i] = params[i] / pow(10, omag)
            dx[i] = (ub[i] - lb[i])/N
    #print(dx)
    lower = params - dx/2
    lower[numpy.where(lower<0.)] = 0.0 #make sure we don't go negative on parameters
    upper = params + dx/2

    return lb, ub, lower, upper

def annealfxn(params, useparams, time, model, envlist, xpdata, xspairlist, lb, ub, norm=False, vardata=False, fileobj=None):
    ''' Feeder function for scipy.optimize.anneal
    '''
    #annlout = scipy.optimize.anneal(pysb.anneal_sundials.annealfxn, paramarr, 
    #                                args=(None, 20000, model, envlist, xpnormdata, 
    #                                [(2,1),(4,2),(7,3)], lb, ub, True, True), 
    #                                lower=lower, upper=upper, full_output=1)
    # sample anneal call full model:
    # params: parameters to be optimized, at their values for the given annealing step
    # lower,upper: arrays from get array function or something similar from getgenparambounds
    # lb, ub: lower bound and upper bound for function from getgenparambounds
    #
    # sample anneal call, optimization of some parameters
    #   annlout = scipy.optimize.anneal(pysb.anneal_sundials.annealfxn, smacprm, args=(smacnum, 25000, model, envlist, xpdata,
    #            [(2,2), (3,3)], lower=lower, upper=upper, full_output=1)
    #
    # sample anneal call, optimization for ALL parameters
    # 
    #

    
    if numpy.greater_equal(params, lb).all() and numpy.less_equal(params, ub).all():
        print("Integrating...")
        outlist = annlodesolve(model, time, envlist, params, useparams)
        # specify that this is normalized data
        if norm is True:
            print("Normalizing data")
            datamax = numpy.max(outlist[0], axis = 1)
            datamin = numpy.min(outlist[0], axis = 1)
            outlistnorm = ((outlist[0].T - datamin)/(datamax-datamin)).T
            # xpdata[0] should be time, get from original array
            outlistnorm[0] = outlist[0][0].copy()
            # xpdata here is normalized, and so is outlistnorm
            objout = compare_data(xpdata, outlistnorm, xspairlist, vardata)
        else:
            objout = compare_data(xpdata, outlist[0], xspairlist, vardata)
    else:
        print("======>VALUE OUT OF BOUNDS NOTED")
        temp = numpy.where((numpy.logical_and(numpy.greater_equal(params, lb), numpy.less_equal(params, ub)) * 1) == 0)
        for i in temp:
            print("======>",i, params[i])
        objout = 1.0e300 # the largest FP in python is 1.0e308, otherwise it is just Inf

    # save the params and temps for analysis
    # FIXME If a parameter is out of bounds, outlist and outlistnorm will be undefined and this will cause an error
    if fileobj:
        if norm:
            writetofile(fileobj, params, outlistnorm, objout)
        else:
            writetofile(fileobj, params, outlist, objout)
    
    return objout

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

# FIXME
# FIXME: THESE FUNCTIONS SHOULD PROBABLY NOT BE INCLUDED IN THE FINAL VERSION OF THE ANNEAL SUNDIALS FUNCTION
# FIXME

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
    obj *= xpsamples
    
    print("OBJOUT-10-90:(%f,%f):%f"%(Tdsim, Tssim, obj))

    return obj    


def annealfxncust(params, useparams, time, model, envlist, xpdata, xspairlist, tenninetylist, lb, ub, norm=False, vardata=False, fileobj = False):
    ''' Feeder function for scipy.optimize.anneal
    '''
    # Customized anneal function for the case when Smac is not fit to a function. Here we
    # measure the Smac output from the model and use a 10-90 criterion to extract Td and Ts.
    # We then use a chi-square comparison to these two values for the Smac contribution to the 
    # data fitting. 
    #
    # 
    #

    if numpy.greater_equal(params, lb).all() and numpy.less_equal(params, ub).all():
        print("Integrating...")
        outlist = annlodesolve(model, time, envlist, params, useparams)
        # specify that this is normalized data
        if norm is True:
            print("Normalizing data")
            datamax = numpy.max(outlist[0], axis = 1)
            datamin = numpy.min(outlist[0], axis = 1)
            outlistnorm = ((outlist[0].T - datamin)/(datamax-datamin)).T
            # xpdata[0] should be time, get from original array
            outlistnorm[0] = outlist[0][0].copy()
            # xpdata here is normalized, and so is outlistnorm
            objout = compare_data(xpdata, outlistnorm, xspairlist, vardata)
            # This takes care of the IC/EC-RP comparisons
            # Now SMAC
            tn = tenninetycomp(outlistnorm, tenninetylist,len(xpdata[0]))
            objout += tn 
            print("objout TOT:", objout)
        else:
            objout = compare_data(xpdata, outlist[0], xspairlist, vardata)
            tn = tenninetycomp(outlistnorm, tenninetylist)
            objout += tn 
    else:
        print("======>VALUE OUT OF BOUNDS NOTED")
        temp = numpy.where((numpy.logical_and(numpy.greater_equal(params, lb), numpy.less_equal(params, ub)) * 1) == 0)
        for i in temp:
            print("======>",i, params[i])
        objout = 1.0e300 # the largest FP in python is 1.0e308, otherwise it is just Inf
    return objout

    





    





