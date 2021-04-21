import numpy as np
import math
import matplotlib.pyplot as plt
import scipy, scipy.stats
import matplotlib.ticker as mticks
#plt.rcParams.update({
#    "text.usetex": True})



# data has format: freq	pk-pk (mv)	uncert pk	phase	uncert phase

data_0damping = np.genfromtxt("0damping400mv.csv",dtype='float',delimiter=',',skip_header=1)
data_100damping = np.genfromtxt("100mAdamping400mv.csv",dtype='float',delimiter=',',skip_header=1)
data_250damping = np.genfromtxt("250mAdamped400mv.csv",dtype='float',delimiter=',',skip_header=1)


#vertical lines
const_x = [17.84,17.84]
y = [0,1200]


#0 damping
freq = data_0damping[:,0]
vel = data_0damping[:,1]
err_vel = data_0damping[:,2]
phase = data_0damping[:,3]
err_phase = data_0damping[:,4]

power = vel**2
err_power = err_vel * 2 #error propergation

figure, ax = plt.subplots(2, 2)

#f vs velocity resonance
figure.suptitle("0 damping")
figure.tight_layout()
ax[0, 0].errorbar(freq, vel, err_vel, fmt = "r.", markersize=1, elinewidth=0.5)
ax[0, 0].set_title("Frequency vs Velocity")

ax[0,0].plot(const_x,y,markersize=0,lw=0.5)

# For phase
ax[0, 1].errorbar(freq, phase, err_phase, fmt = "r.", markersize=1, elinewidth=0.5)
ax[0, 1].set_title("Frequency vs Phase")


#for power
#power proportiona to vel^2
ax[1,0].errorbar(freq, power, err_power, fmt = "r.", markersize=1, elinewidth=0.5)
ax[1, 0].set_title("Frequency vs Power")

ax[1,1].set_axis_off()

plt.savefig('0damp.png', dpi=800)
plt.show()



#freq vs vel, seperate
plt.errorbar(freq, vel, err_vel,fmt = "rx", markersize=1, elinewidth=0.5)
plt.plot(const_x,y,markersize=0,lw=0.5)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (v)")
plt.title("Frequency vs Amplitude")
plt.savefig('0damp_1.png', dpi=800)
#plt.xlim((13.5, 20.5))
#plt.xaxis.set_major_locator(mticks.MultipleLocator(0.5))
plt.show()


#freq vs power seperate
p_max = max(power)
print(p_max)
p_max_2 = [p_max/2,p_max/2]
x = [17.3,18.4]

const_x = [17.83,17.83]
y = [0,1145000]

fig, ax = plt.subplots(1,1)
fig.suptitle("0 damping Power resonance")
plt.grid(True, which='both' ,linewidth=0.25, color='#808080', linestyle='-')
ax.set_xlim((17.3, 18.4))
ax.xaxis.set_major_locator(mticks.MultipleLocator(0.1))
ax.yaxis.set_major_locator(mticks.MultipleLocator(100000))
ax.errorbar(freq, power, err_power, fmt = "r.", markersize=2, elinewidth=1)
ax.set_title("Frequency vs Power")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (v^2)")
ax.plot(x,p_max_2,markersize=0,lw=0.5)
ax.plot(const_x,y,markersize=0,lw=0.5)
plt.savefig('0dampPower.png', dpi=800)
plt.show()



#freq vs phase, seperate
plt.errorbar(freq, phase, err_phase,fmt = "rx", markersize=2, elinewidth=1)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (deg)")
plt.title("Frequency vs Phase")
plt.savefig('0damp_2.png', dpi=800)
plt.show()


#### BY EYE
f_fwhm_range = [7.79,7.88]
f_fwhm = (7.88 - 7.79) / 2
print("f_fwhm = ", f_fwhm)

# Q = w_0 / gamma = f_0 / f_fwhm

f_0 = 17.83 ######################################NEED TO GET f_0 FROM DAY 2 EXPERIMENT (and gamma) ###########################################################################################

Q = 17.83 / f_fwhm

print("Q factor = ", Q)









#100 damping
#vertical lines
const_x = [17.83,17.83]
y = [0,700]


#0 damping
freq = data_100damping[:,0]
vel = data_100damping[:,1]
err_vel = data_100damping[:,2]
phase = data_100damping[:,3]
err_phase = data_100damping[:,4]

power = vel**2
err_power = err_vel * 2 #error propergation

figure, ax = plt.subplots(2, 2)

#f vs velocity resonance
figure.suptitle("100 damping")
figure.tight_layout()
ax[0, 0].errorbar(freq, vel, err_vel, fmt = "r.", markersize=1, elinewidth=0.5)
ax[0, 0].set_title("Frequency vs Velocity")

ax[0,0].plot(const_x,y,markersize=0,lw=0.5)

# For phase
ax[0, 1].errorbar(freq, phase, err_phase, fmt = "r.", markersize=1, elinewidth=0.5)
ax[0, 1].set_title("Frequency vs Phase")


#for power
#power proportiona to vel^2
ax[1,0].errorbar(freq, power, err_power, fmt = "r.", markersize=1, elinewidth=0.5)
ax[1, 0].set_title("Frequency vs Power")

ax[1,1].set_axis_off()

plt.savefig('100damp.png', dpi=800)
plt.show()



#freq vs vel, seperate
plt.errorbar(freq, vel, err_vel,fmt = "rx", markersize=1, elinewidth=0.5)
plt.plot(const_x,y,markersize=0,lw=0.5)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (v)")
plt.title("Frequency vs Amplitude")
plt.savefig('100damp_1.png', dpi=800)
#plt.xlim((13.5, 20.5))
#plt.xaxis.set_major_locator(mticks.MultipleLocator(0.5))
plt.show()


#freq vs power seperate
p_max = max(power)
print(p_max)
p_max_2 = [p_max/2,p_max/2]
x = [17.3,18.4]

const_x = [17.83,17.83]
y = [0,400000]
const_x_damped = [17.84,17.84]
y = [0,400000]

fig, ax = plt.subplots(1,1)
fig.suptitle("100 damping Power resonance")
plt.grid(True, which='both' ,linewidth=0.25, color='#808080', linestyle='-')
ax.set_xlim((17.3, 18.4))
ax.xaxis.set_major_locator(mticks.MultipleLocator(0.1))
ax.yaxis.set_major_locator(mticks.MultipleLocator(100000))
ax.errorbar(freq, power, err_power, fmt = "r.", markersize=2, elinewidth=1)
ax.set_title("Frequency vs Power")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (v^2)")
ax.plot(x,p_max_2,markersize=0,lw=0.5)
ax.plot(const_x,y,markersize=0,lw=0.5)
ax.plot(const_x_damped,y,markersize=0,lw=0.5)
plt.savefig('100dampPower.png', dpi=800)
plt.show()



#freq vs phase, seperate
plt.errorbar(freq, phase, err_phase,fmt = "rx", markersize=2, elinewidth=1)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (deg)")
plt.title("Frequency vs Phase")
plt.savefig('100damp_2.png', dpi=800)
plt.show()








#250 damping
#vertical lines
const_x = [17.83,17.83]
y = [0,260]


#250 damping
freq = data_250damping[:,0]
vel = data_250damping[:,1]
err_vel = data_250damping[:,2]
phase = data_250damping[:,3]
err_phase = data_250damping[:,4]

power = vel**2
err_power = err_vel * 2 #error propergation

figure, ax = plt.subplots(2, 2)

#f vs velocity resonance
figure.suptitle("250 damping")
figure.tight_layout()
ax[0, 0].errorbar(freq, vel, err_vel, fmt = "r.", markersize=1, elinewidth=0.5)
ax[0, 0].set_title("Frequency vs Velocity")

ax[0,0].plot(const_x,y,markersize=0,lw=0.5)

# For phase
ax[0, 1].errorbar(freq, phase, err_phase, fmt = "r.", markersize=1, elinewidth=0.5)
ax[0, 1].set_title("Frequency vs Phase")


#for power
#power proportiona to vel^2
ax[1,0].errorbar(freq, power, err_power, fmt = "r.", markersize=1, elinewidth=0.5)
ax[1, 0].set_title("Frequency vs Power")

ax[1,1].set_axis_off()

plt.savefig('250damp.png', dpi=800)
plt.show()



#freq vs vel, seperate
plt.errorbar(freq, vel, err_vel,fmt = "rx", markersize=1, elinewidth=0.5)
plt.plot(const_x,y,markersize=0,lw=0.5)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (v)")
plt.title("Frequency vs Amplitude")
plt.savefig('250damp_1.png', dpi=800)
#plt.xlim((13.5, 20.5))
#plt.xaxis.set_major_locator(mticks.MultipleLocator(0.5))
plt.show()


#freq vs power seperate
p_max = max(power)
print(p_max)
p_max_2 = [p_max/2,p_max/2]
x = [17.3,18.4]

const_x = [17.83,17.83]
y = [0,60000]
const_x_damped = [17.855,17.855]
y = [0,60000]

fig, ax = plt.subplots(1,1)
fig.suptitle("250 damping Power resonance")
plt.grid(True, which='both' ,linewidth=0.25, color='#808080', linestyle='-')
ax.set_xlim((17.0, 18.8))
ax.xaxis.set_major_locator(mticks.MultipleLocator(0.2))
ax.yaxis.set_major_locator(mticks.MultipleLocator(10000))
ax.errorbar(freq, power, err_power, fmt = "r.", markersize=2, elinewidth=1)
ax.set_title("Frequency vs Power")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (v^2)")
ax.plot(x,p_max_2,markersize=0,lw=0.5)
ax.plot(const_x,y,markersize=0,lw=0.5)
ax.plot(const_x_damped,y,markersize=0,lw=0.5)
plt.savefig('250dampPower.png', dpi=800)
plt.show()



#freq vs phase, seperate
plt.errorbar(freq, phase, err_phase,fmt = "rx", markersize=2, elinewidth=1)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (deg)")
plt.title("Frequency vs Phase")
plt.savefig('250damp_2.png', dpi=800)
plt.show()









################################## DAY 2 ##################################


data_2 = np.genfromtxt("day2.csv",dtype='float',delimiter=',',skip_header=0)
time = data_2[:,0]
vel = data_2[:,1]


times_max = []
peaks = []

#peak finding algorithm
for x in range(0, len(vel)-1):
    if vel[x] >= vel[x-1] and vel[x] > vel[x+1] and vel[x] > 0:
        times_max.append(time[x])
        peaks.append(vel[x])
        
#print(max_values)
    
#plot raw data then add x's @ the peaks to show the peak finding algorithm works
plt.plot(time, vel)
plt.plot(times_max, peaks, "r.", markersize=1)
plt.title("Raw data for an Undamped Free Oscillator")
plt.xlabel("time (s)")
plt.ylabel("Voltage (v)")
plt.savefig("time vs amplitude.png",dpi=800)
plt.show()

#print(type(times_max))
times_max = np.array(times_max)

def f(t, gamma, A):
    return A * np.exp(-(gamma * t)/2)


# perform the fit
params, cv = scipy.optimize.curve_fit(f, times_max, peaks)
gamma, A = params


# plot the results
plt.plot(times_max, peaks, 'rx', label="data")
plt.plot(times_max, f(times_max, gamma, A), '-', label="fitted")
plt.title("Fitted Exponential Curve For Undamped Free Oscillator")
plt.xlabel("time (s)")
plt.ylabel("Voltage (v)")
plt.text(0.1, 0.1, f"Y = {A} * e**(-({gamma} * t)/2)")
plt.savefig("time vs peaks exp",dpi=800)
plt.show()

# inspect the parameters
print(f"Y = {A} * e**(-({gamma} * t)/2)")


def g(x,m,c):
    return m*x+c

params2, cv = scipy.optimize.curve_fit(g, times_max, np.log(peaks))
m, c = params2
print(f"ln(Y) = {m}t + {c}")

plt.plot(times_max, np.log(peaks), 'rx')
plt.plot(times_max, g(times_max, m, c))
plt.title("Fitted Linear Curve for undamped free oscillator")
plt.xlabel("time (s)")
plt.ylabel("ln(Voltage)")
plt.text(0.1, -2.3,f"ln(Y) = {m}t + {c}")
plt.savefig("time vs peaks lin",dpi=800)
plt.show()


gamma_calc = (gamma + (-m * 2))/2
print("gamma (free oscillation) ",gamma_calc)
error_on_gamma = gamma - gamma_calc
print("error_on_gamma: ", error_on_gamma)
f_0 = 17.93
w_0 = (2*np.pi)*f_0
Q_factor = (w_0 / gamma_calc)

print("Q factor (free oscillator): ", Q_factor)


#times between peaks

delta_ts = []
for x in range(0,len(times_max)):
    try:
        delta_ts.append(times_max[x+1]-times_max[x])
    except:
        pass
#print(delta_ts)


#times_max2 = []
#for x in range(0,len(delta_ts)):
#    times_max2.append(times_max[x])

#plt.plot(times_max2, delta_ts)
#plt.show()

average_T = sum(delta_ts)/len(delta_ts)
estimated_f_0 = 1/average_T

print("Average time period: ", average_T," esitmate f_0: ",estimated_f_0)

#Time period seems to be arround 0.056
# f = 1/0.056 = 17.86!
