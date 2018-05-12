import numpy as np
import matplotlib.pyplot as plt

# cos parameters
a = 3.
c = 1.
phi = 0.
f = 2.
T = 2 * np.pi

# gaussian parameters
A = 5.
B = 10.
L = 1.

# fft parameters
N = 500
freq = np.fft.fftfreq(N, 0.002)

# cos
t = np.linspace(0, T, N)
cos = a * np.cos(f*t + phi) + c

# gaussian
x = np.linspace(-L, 2*L, N)
gaussian = A * np.exp(-B*(x-L/2)**2)

# fft gaussian analytical
g_an = np.sort(A/(np.sqrt(2*B)) * np.exp(freq*(freq-2*1j*B*L)/(4*B)))

# fft for cos
cos_fft = np.fft.fft(cos) / N

# fft for gaussian
g_fft = A * np.fft.fft(gaussian) / N

# double fft
cos_2fft = np.fft.fft(cos_fft) * a

# plot
fig, ((cos_plt, cos_fft_plt, cos_2fft_plt), (g_plt, g_fft_plt, g_an_plt)) = plt.subplots(2, 3)

cos_plt.plot(t, cos)
cos_plt.set_ylabel('$f(t)$')
cos_plt.set_xlabel('$t$')

cos_fft_plt.plot(freq, np.abs(cos_fft))
cos_fft_plt.set_ylabel('$F[f(t)]$')
cos_fft_plt.set_xlabel('$k$')
cos_fft_plt.set_xlim(-5, 5)

cos_2fft_plt.plot(t, cos_2fft)
cos_2fft_plt.set_xlabel('$t$')
cos_2fft_plt.set_ylabel('$F^{-1}[F[f(t)]]$')

g_plt.plot(x, gaussian)
g_plt.set_ylabel('$f(x)$')
g_plt.set_xlabel('$x$')

g_fft_plt.plot(freq, np.abs(g_fft))
g_fft_plt.set_ylabel('$F[f(x)]$')
g_fft_plt.set_xlabel('$k$')

g_an_plt.plot(freq, np.abs(g_an))
g_an_plt.set_ylabel('$F[f(x)]$ (analytical)')
g_an_plt.set_xlabel('$x$')

plt.tight_layout()
plt.show()

