import pymc
import scipy as sp
import matplotlib.pyplot as plt

def coin_uniform(data):
    '''
    Statistical model for coin-flipping with a uniform prior.

    Arguments
        data: array of coin-flips. 0 means a tail, 1 means a head.

    Returns
        a pymc Model object.
    '''
    # Uniform Prior
    H = pymc.Uniform('H (uniform prior)', lower=0, upper=1)

    # Likelihood
    L = pymc.Bernoulli('L', p=H, value=data, observed=True)
    
    return pymc.Model([H, L])

def coin_gaussian(data, m=0.5, s=0.2):
    '''
    Statistical model for coin-flipping with a Gaussian prior.

    Arguments
        data: array of coin-flips. 

    Returns
        a pymc Model object.
    '''
    # Gaussian prior
    H = pymc.Normal('H (gaussian prior)', mu=m, tau=1/s**2)

    # Likelihood
    L = pymc.Bernoulli('L', p=H, value=data, observed=True)
    
    return pymc.Model([H, L])

def change_iter(data, iters):
    '''
    Runs the coin-flipping model for different numbers of chains and plots the 
    graphs.

    Arguments
        data: array of coin-flips. 
        iters: list of chain lengths.
    '''
    for i, n in enumerate(iters):
        u = pymc.MCMC(coin_uniform(data))
        u.sample(iter=n)
        u_trace = u.trace('H (uniform prior)')[:]

        g = pymc.MCMC(coin_gaussian(data))
        g.sample(iter=n)
        g_trace = g.trace('H (gaussian prior)')[:]

        u.summary()
        g.summary()

        # Plot histogram
        # Uniform prior
        plt.figure(1)
        plt.suptitle('H (uniform prior)', fontsize=13)
        plt.subplot(1, 3, i+1)
        plt.hist(u_trace, bins=100)
        plt.title('iterations = {}'.format(n))
        plt.xlabel('H')
        plt.ylabel('frequency')
        
        # Gaussian prior
        plt.figure(2)
        plt.suptitle('H (gaussian prior)', fontsize=13)
        plt.subplot(1, 3, i+1)
        plt.hist(g_trace, bins=100)
        plt.title('iterations = {}'.format(n))
        plt.xlabel('H')
        plt.ylabel('frequency')

def change_mu(data, mus):
    '''
    Runs the coin-flipping model for different values of biases for the Gaussian
    prior. Plots the graphs. 

    Arguments
        data: array of coin-flips. 
        mus: list of coin bias values.
    '''
    plt.figure()
    
    for i, mu in enumerate(mus):
        g = pymc.MCMC(coin_gaussian(data, m=mu))
        g.sample(iter=10000, thin=10)
        g_trace = g.trace('H (gaussian prior)')[:]
        g.summary()

        # Plot histogram
        plt.subplot(2, 2, i+1)
        plt.hist(g_trace, bins=100)
        plt.title('$\mu$ = {}'.format(mu))
        plt.xlabel('H')
        plt.ylabel('frequency')
    
    plt.tight_layout()

def change_sigma(data, sigmas):
    '''
    Runs the coin-flipping model for different values of sigma for the Gaussian
    prior. Plots the graphs. 

    Arguments
        data: array of coin-flips. 
        sigmas: list of sigma values for the Gaussian prior.
    '''
    plt.figure()
    
    for i, sigma in enumerate(sigmas):
        g = pymc.MCMC(coin_gaussian(data, s=sigma))
        g.sample(iter=10000, thin=10)
        g_trace = g.trace('H (gaussian prior)')[:]
        g.summary()
       
        # Plot histogram
        plt.subplot(2, 2, i+1)
        plt.hist(g_trace, bins=100)
        plt.title('$\sigma$ = {}'.format(sigma))
        plt.xlabel('H')
        plt.ylabel('frequency')
    
    plt.tight_layout()

# Params
N = 1000
true_h = 0.30
iters = [100, 1000, 10000]
chains = [2, 3, 4]
mus = [0.20, 0.40, 0.60, 0.80]
sigmas = [0.10, 0.30, 0.50, 0.70]

# Draw number of heads from a binomial distribution. 0 means a tail, 1 means a 
# head. 
dat = sp.stats.bernoulli.rvs(true_h, size=N)

change_iter(dat, iters)
change_mu(dat, mus)
change_sigma(dat, sigmas)

plt.show()
