
###################################################################
#  Conductance analysis (VmT)                                     #
#                                                                 #
#  based on the publication                                       #
#                                                                 #
#  Martin Pospischil, Zuzanna Piwkowska, Thierry Bal and          #
#  Alain Destexhe "Extracting Synaptic Conductances from Single   #
#  Membrane Potential Traces", Neuroscience 158: 545-552, 2009.   #
#                                                                 #
###################################################################

The application extracts the parameters of the distributions of 
excitation and inhibition from a single current clamp recording, 
based on a maximum likelihood method. The application is run on 
pieces of the voltage trace that are delimited by spikes or a 
maximal length. As output, for each interval a line is written to 
the output file, containing the number of datapoints and the 
distribution parameters (in uS) of the current run as well as the 
momentary average. The application is started by invoking the 
command 'python VmT.py' from the system prompt. In order for this 
to work, the python scripting language along with the packages  
'Numeric' and 'pylab' need to be installed.

Adjustable parameters are given in the header.py file and comprise:

vmFile      -   input file name containing the voltage time course
                as a single column
resfile     -   output file name

Iext        -   constant injected current in nA
gtot        -   total input conductance in uS
C           -   capacitance in nF
gl          -   leak conductance in uS
Vl          -   leak reversal potential in mV
Ve          -   reversal potential of excitation in mV
Vi          -   reversal potential of inhibition in mV
te          -   correlation time constant of excitation in ms
ti          -   correlation time constant of inhibition in ms



vt          -   threshold for spike detection in mV
dt          -   time between two datapoints in ms
t_pre       -   excluded time preceding spike
t_post      -   excluded time after spike

n_smooth    -   SD in timesteps of the Gaussian filter that is used 
                for smoothing
n_ival      -   max nb of intervals analysed

n_minISI    -   min nb of datapoints in interval
n_maxISI    -   max nb of datapoints in interval

g_start     -   starting point for minimisation [ge,se,si] in uS
