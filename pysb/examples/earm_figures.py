from pysb.integrate import odesolve
from pylab import *

from earm_1_0 import model


# saturating level of ligand (corresponding to ~1000 ng/ml SuperKiller TRAIL)
Lsat = 6E4; 

# relationship of ligand concentration in the model (in # molecules/cell) to actual TRAIL concentration (in ng/ml)
Lfactor = model.parameters['L_0'].value / 50;

L_0_baseline = model.parameters['L_0'].value


def fig_4a():
    t = linspace(0, 20*3600, 20*60+1)  # 20 hours, in seconds, 1 min sampling
    dt = t[1] - t[0]

    Ls_exp = Lsat / array([1, 4, 20, 100, 500])
    Td_exp = [144.2, 178.7, 236,   362.5, 656.5]
    Td_std = [32.5,   32.2,  36.4,  78.6, 171.6]
    Ts_exp = [21.6,   23.8,  27.2,  22.0,  19.0]
    Ts_std = [9.5,     9.5,  12.9,   7.7,  10.5]

    CVenv = 0.2
    # num steps was originally 40, but 15 is plenty smooth enough for screen display
    Ls = floor(logspace(1,5,15)) 

    fs = empty_like(Ls)
    Ts = empty_like(Ls)
    Td = empty_like(Ls)
    for i in range(len(Ls)):
        model.parameters['L_0'].value = Ls[i]

        print "integrating model"
        x = odesolve(model, t)

        fs[i] = (x['PARP'][0] - x['PARP'][-1]) / x['PARP'][0]
        dP = 60 * (x['PARP'][:-1] - x['PARP'][1:]) / (dt * x['PARP'][0])  # in minutes
        ttn = argmax(dP)
        dPmax = dP[ttn]
        Ts[i] = 1 / dPmax  # minutes
        Td[i] = t[ttn] / 60  # minutes

    figure()
    plot(Ls/Lfactor, Td, 'g-', Ls/Lfactor, (1-CVenv)*Td, 'g--', Ls/Lfactor, (1+CVenv)*Td, 'g--')
    errorbar(Ls_exp/Lfactor, Td_exp, Td_std, None, 'go', capsize=0),
    ylabel('Td (min)'),
    xlabel('TRAIL (ng/ml)'),
    a = gca()
    a.set_xscale('log')
    a.set_xlim((min(Ls) / Lfactor, max(Ls) / Lfactor))
    a.set_ylim((0, 1000))

    model.parameters['L_0'].value = L_0_baseline


def fig_4b():
    t = linspace(0, 6*3600, 6*60+1)  # 6 hours
    x = odesolve(model, t)

    x_norm = array([x['Bid'], x['PARP'], x['mSmac']]).T
    x_norm = 1 - x_norm / x_norm[0, :]  # gets away without max() since first values are largest

    # this is what I originally thought 4B was plotting.  oddly it's very close.
    #x_norm = array([x['tBid'], x['CPARP'], x['cSmac']]).T
    #x_norm /= x_norm.max(0)

    tp = t / 3600  # x axis as hours

    figure()
    plot(tp, x_norm[:,0], 'b', label='IC substrate (tBid)')
    plot(tp, x_norm[:,1], 'y', label='EC substrate (cPARP)')
    plot(tp, x_norm[:,2], 'r', label='MOMP (cytosolic Smac)')
    legend(loc='upper left', bbox_to_anchor=(0,1)).draw_frame(False)
    xlabel('Time (hr)')
    ylabel('fraction')
    a = gca()
    a.set_ylim((-.05, 1.05))


fig_4a()
fig_4b()
show()
