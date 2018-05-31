import pymc
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

def lighthouse(data):
    # Uniform priors 
    x = pymc.Uniform('x', lower=0, upper=10)
    y = pymc.Uniform('y', lower=0, upper=10)

    # Likelihood
    L = pymc.Cauchy('L', alpha=x, beta=y, value=data, observed=True)

    return pymc.Model([x, y, L])

def change_iter(data, iters, true_x, true_y):
    '''
    Runs the lighthouse model for different numbers of chains and plots the 
    graphs.

    Arguments
        data: array of flash positions measured along the shore.
        iters: list of chain lengths.
    '''
    plt.figure(3)
    
    for i, n in enumerate(iters):
        l = pymc.MCMC(lighthouse(data))
        l.sample(iter=n)
        x = l.trace('x')[:]
        y = l.trace('y')[:]
        
        # Plot histogram
        plt.subplot(2, 2, i+1)
        plt.hist2d(x, y, bins=100, range=[[true_x-1.2, true_x+1.2], [true_y-1.2, true_y+1.2]], cmap=plt.cm.get_cmap('magma'))
        plt.title('iterations = {}'.format(n))
        plt.xlabel('x')
        plt.ylabel('y')

    plt.tight_layout()

def max_xy(data, iters, x_true, y_true):
    '''
    Computes the most probable position of the lighthouse separately in x and y
    as a function of chain length. Plots the graphs. 

    Arguments
        data: array of flash positions measured along the shore.
        iters: list of chain lengths.
        x_true: true x value of lighthouse.
        y_true: true y value of lighthouse.
    '''
    x, y, fx, fy = np.array([]), np.array([]), np.array([]), np.array([])
    
    for i, n in enumerate(iters):
        l = pymc.MCMC(lighthouse(data))
        l.sample(iter=n)
        x_trace = l.trace('x')[:]
        y_trace = l.trace('y')[:]
        
        countx, binx = np.histogram(x_trace, bins=100)
        county, biny = np.histogram(y_trace, bins=100)
        maxx = np.argmax(countx)
        maxy = np.argmax(county)
        avgx = (binx[maxx] + binx[maxx-1])/2.
        avgy = (biny[maxy] + biny[maxy-1])/2.
        x = np.append(x, avgx)
        y = np.append(y, avgy)
        fx = np.append(fx, np.max(countx))
        fy = np.append(fy, np.max(county))
    
    plt.subplot(1, 2, 1)
    plt.scatter(x, fx)
    plt.axvline(x=x_true, color='gray')
    plt.xlabel('x')
    plt.ylabel('frequency')

    for i, n in enumerate(iters):
        plt.annotate(n, (x[i], fx[i]))

    plt.subplot(1, 2, 2)
    plt.scatter(y, fy)
    plt.axvline(x=y_true, color='gray')
    plt.xlabel('y')
    plt.ylabel('frequency')

    for i, n in enumerate(iters):
        plt.annotate(n, (y[i], fy[i]))
    
    plt.ylim(ymin=0)
    plt.tight_layout()
    plt.show()
    
def interloper(data, alpha, beta):
    '''
    Simulates the effect of an interloper - the case where a lighthouse ship comes
    nearby the lighthouse and sends out its own light pulses.

    Arguments
        data: array of flash positions measured along the short from the lighthouse.
        alpha: distance of the ship along the shore.
        beta: distance of the ship out to sea. 

    Returns
        modified array with flash positions measured from the ship embedded within
        a copy of the array of flashes from the lighthouse. 
    '''
    dat = data[:]
    
    # Draw flashes from a Cauchy distribution
    flash = sp.cauchy.rvs(loc=alpha, scale=beta, size=len(data))
    
    for item in flash:
        ind = np.random.randint(0, len(dat))
        dat = np.insert(dat, ind, item)
    
    return dat

# Params
N = 1000
true_x = 2
true_y = 2
iters = [100, 1000, 5000, 10000]

# Draw flashes from a Cauchy distribution
dat = sp.cauchy.rvs(loc=true_x, scale=true_y, size=N)

# Interloper
ship = interloper(dat, 3, 3)

#max_xy(dat, iters, true_x, true_y)
change_iter(dat, iters, true_x, true_y)

# Simulate models
l = pymc.MCMC(lighthouse(dat))
l.sample(iter=10000)
l.summary()
x = l.trace('x')[:]
y = l.trace('y')[:]

loper = pymc.MCMC(lighthouse(ship))
loper.sample(iter=10000)
loper.summary()
xl = loper.trace('x')[:]
yl = loper.trace('y')[:]

# Plot
plt.figure(1)
plt.hist2d(x, y, bins=100, range=[[true_x-1.5, true_x+1.5], [true_y-1.5, true_y+1.5]], cmap=plt.cm.get_cmap('magma'))
plt.title('Without Interloper')
plt.xlabel('x')
plt.ylabel('y')

plt.figure(2)
plt.hist2d(xl, yl, bins=100, range=[[true_x-1.5, true_x+1.5], [true_y-1.5, true_y+1.5]], cmap=plt.cm.get_cmap('magma'))
plt.title('With Interloper')
plt.xlabel('x')
plt.ylabel('y')

plt.show()
