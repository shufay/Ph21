import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

def const(num):
    '''
    A uniform distribution over all H.

    Argument
        num: Number of H values.

    Returns
        An array of equal probabilities for each H.
    '''
    return 1./num * np.ones(num, dtype=np.float64)

def gaussian(H, mu, sigma):
    '''
    A Gaussian distribution over all H.

    Argument
        H: An array of H values.
        mu: Mean of the Gaussian distribution.
        sigma: Standard deviation of the Gaussian distribution.

    Returns
        An array of probabilities computed from the Gaussian distribution for each
        H.
    '''
    power = -np.power((H-mu), 2.) / (2. * np.power(sigma, 2.))
    return 1./np.sqrt(2.*np.pi*sigma) * np.exp(power)

def sim(N, true_h, prior_dist, mu=0.5, sigma=0.2):
    '''
    Runs the coin-flipping simulation.

    Argument
         N: Number of trials.
         true_h: The 'true' value of H for the coin.
         prior_dist: String specifying which prior to be used.
         mu: Mean of the Gaussian distribution.
         sigma: Standard deviation of the Gaussian distribution.

    Returns
        An array of H values and posterior probabilities for each H.
    '''
    H = np.linspace(0.0, 1.0, 500)
    heads = 0.
    
    if prior_dist.lower() == 'uniform':
        prior = const(len(H))
    
    else: 
        prior = gaussian(H, mu, sigma)

    for n in range(1, N+1):
        flip = np.random.random()
        
        if flip < true_h:
            heads += 1.
        
        likelihood = sp.comb(n, heads) * np.power(H, heads) * np.power((1-H), (n-heads))
        posterior = likelihood * prior
        prior = posterior
    
    prior = prior / prior.sum()
    return H, prior

def plot(N, true_h, prior_dist, mu=0.5, sigma=0.5):
    '''
    Plots the results from sim (able to do so for multiple number of trials).

    Argument
        N: array of the number of trials.
        true_h: The 'true' value of H.
        prior_dist: String specifying which prior to be used.
        mu: Mean of the Gaussian distribution.
        sigma: Standard deviation of the Gaussian distribution.
    '''
    plt.figure()
    plt.suptitle('{}'.format(prior_dist))

    for i, n in enumerate(N):
        H, res = sim(n, true_h, prior_dist, mu, sigma)
        plt.subplot(4, 4, i+1)
        plt.plot(H, res)
        plt.title('n = {}'.format(n))
        plt.xlabel('H')
        plt.ylabel('P(H|data)')
        print(n)
    
    plt.tight_layout()
    plt.show()

# run
#N = 10
N1 = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
N2 = [256, 512, 1024, 2048]
true_h = 0.25

plot(N1, true_h, 'uniform')
plot(N1, true_h, 'gaussian')

'''
H, posterior, prior_dist = sim(N, true_h, 'gaus', mu=0.7)
plt.plot(H, posterior, prior_dist)
plt.show()
'''
