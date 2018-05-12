import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

with open('data/arecibo1.txt', 'r') as arecibo1:
    arecibo1_dat = np.array([])
    line = arecibo1.readline()

    while line != '':
        arecibo1_dat = np.append(arecibo1_dat, float(line[:-1]))
        line = arecibo1.readline()

# gaussian * sin envelope
def gaussian(t, t0, tc, A, f, phi):
    return np.exp(-(t-t0)**2/tc**2) * A*np.sin(f*t + phi)

# arecibo parameters
N = 32768
dt = 0.001
t = np.linspace(0.0, N*dt, N)

# fft
f = np.fft.fftfreq(N, 0.001)
#freq = np.fft.fftshift(f)
arecibo1_fft = np.fft.fft(arecibo1_dat) / N

# signal frequency
sig_max = np.argmax(arecibo1_fft)
sig_f = abs(f[sig_max])
print('Signal Frequency: {} Hz'.format(sig_f))

# fit curve
popt, pcov = curve_fit(gaussian, t, arecibo1_dat)

# time constant
print('Time constant = {} s'.format(popt[0]))

# model fft
gaussfit = gaussian(t, popt[0], popt[1], popt[2], popt[3], popt[4])
gaussfit_fft = np.fft.fft(gaussfit) 


# plot
fig, (a1, a1_fft, a1_fitted) = plt.subplots(3, 1)
a1.plot(t, arecibo1_dat)
a1.set_title('Output from Arecibo 1')
a1.set_xlabel('t / s')
a1.set_ylabel('Output')

a1_fft.plot(f, np.abs(arecibo1_fft))
a1_fft.set_title('Fourier Transform of Output')
a1_fft.set_xlabel('f / Hz')
a1_fft.set_ylabel('Fourier Transform')

a1_fitted.plot(f+sig_f, np.abs(gaussfit_fft))
a1_fitted.set_title('Gaussian x Sinusoid Model')
a1_fitted.set_xlabel('f / Hz')
a1_fitted.set_ylabel('Fourier Transform')

plt.tight_layout()
plt.show()



