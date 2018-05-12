import astropy.stats as ast
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

with open('data/arecibo1.txt', 'r') as arecibo1:
    arecibo1_dat = np.array([])
    line = arecibo1.readline()

    while line != '':
        arecibo1_dat = np.append(arecibo1_dat, float(line[:-1]))
        line = arecibo1.readline()

# gaussian parameters
A = 5.
B = 10.
L = 1.
N = 100

# gaussian
x = np.linspace(-L, L, N)
gaussian = A * np.exp(-B*(x)**2)

f_gauss, p_gauss = ast.LombScargle(x, gaussian).autopower(minimum_frequency=0., maximum_frequency=3.)

# arecibo 1 parameters
n = 32768
dt = 0.001

# arecibo 1
t = np.linspace(0., n*dt, n)
f_a1, p_a1 = ast.LombScargle(t, arecibo1_dat).autopower(minimum_frequency=0., maximum_frequency=200.)

# her x-1
time, mag = np.loadtxt('data/her-x-1.txt',delimiter=' ', unpack=True, skiprows=1)
f_x1, p_x1 = ast.LombScargle(time, mag).autopower()

# plot
fig, ((g, lomb_g), (a1, lomb_a1), (her_x1, lomb_x1)) = plt.subplots(3, 2)

g.plot(x, gaussian, 'b+')
g.set_xlabel('x')
g.set_ylabel('f(x)')
g.set_title('Gaussian')

lomb_g.plot(f_gauss, p_gauss)
lomb_g.set_xlabel('f')
lomb_g.set_ylabel('Power')
lomb_g.set_title('Periodogram')

a1.plot(t, arecibo1_dat)
a1.set_xlabel('t / ms')
a1.set_ylabel('Signal')
a1.set_title('Arecibo 1')

lomb_a1.plot(f_a1, p_a1)
lomb_a1.set_xlabel('f / 1/s')
lomb_a1.set_ylabel('Power')
lomb_a1.set_title('Periodogram')

her_x1.scatter(time, mag)
her_x1.set_xlabel('days / MJD')
her_x1.set_ylabel('Magnitude')
her_x1.set_title('Her X-1')

lomb_x1.plot(1./f_x1, p_x1)
lomb_x1.set_xlabel('period / days')
lomb_x1.set_ylabel('Power')
lomb_x1.set_xlim(0., 10.)
lomb_x1.set_title('Periodogram')

plt.tight_layout()
plt.show()

