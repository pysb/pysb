import numpy as np
import matplotlib.pyplot as plt
import sys
import gillespy
from pysb.examples.tyson_oscillator import model 
from pysb.tools.gillespy_converter import convert as pysb_to_gillespy

gill_tyson = pysb_to_gillespy(model)
number_of_trajectories=10
tyson_trajectories = gillespy.StochKitSolver.run(gill_tyson,t=50,\
                increment=0.01,number_of_trajectories=number_of_trajectories)


from matplotlib import gridspec
gs = gridspec.GridSpec(1,1)
ax0 = plt.subplot(gs[0,0])
time = np.array(tyson_trajectories[0][:,0]) 
S_trajectories = np.array([tyson_trajectories[i][:,1] for i in xrange(number_of_trajectories)]).T

#plot individual trajectories
ax0.plot(time, S_trajectories, 'gray', alpha = 0.1)
#plot mean
ax0.plot(time, S_trajectories.mean(1), 'k--', label = "Mean")
#plot min-max
ax0.plot(time,S_trajectories.min(1), 'b--', label = "Minimum")
ax0.plot(time,S_trajectories.max(1), 'r--', label = "Maximum")
#regression = np.polyfit(time, S_trajectories.mean(1), 1)
#ax0.plot(time,regression[0]*time+regression[1],'o',label =str(regression[1]))
ax0.legend()
ax0.set_xlabel('Time')
ax0.set_ylabel('Species S Count')
plt.tight_layout()
plt.show()
