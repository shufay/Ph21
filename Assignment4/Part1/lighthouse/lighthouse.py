import numpy as np
import matplotlib.pyplot as plt

def cauchy(x, alpha, beta):
    '''
    A Cauchy distribution over all x.

    Arguments
        x: x position along sea shore.
        alpha: distance of lighthouse along the shore.
        beta: distance of lighthouse out to sea.

    Returns
        An array of probabilities for x calculated with the Cauchy distribution.
    '''
    return np.power(beta, 2.) + np.power((x - alpha), 2.)

def sim_alpha(N, true_alpha, true_beta):
    '''
    Runs the lighthouse flashing simulation with known beta.

    Arguments
        N: Number of flashes.
        true_alpha: The true distance of the lighthouse along the shore.
        true_beta: The true distance of the lighthouse out to sea.

    Returns
        An array of alpha values, posterior probabilities for each x and the mean
        x position of flashes detected. 
    '''
    alpha = np.linspace(0., 10., 500)
    log_prior = np.zeros(len(alpha))
    xk = np.array([])

    for n in range(N):
        xk = np.array([])
        flash = np.random.random()
        angle = flash * np.pi
        
        if angle > np.pi / 2:
            angle -= np.pi 

        x = true_beta * np.tan(angle) + true_alpha
        xk = np.append(xk, x)
        log_posterior = log_prior + np.log(true_beta / cauchy(x, alpha, true_beta))
        log_prior = log_posterior
 
    if N != 0:
        log_prior -= log_prior.max()
    
    prior = np.exp(log_prior)
    return alpha, prior, xk.mean()

def sim_alpha_beta(N, true_alpha, true_beta):
    '''
    Runs the lighthouse simulation with alpha and beta unknown.

    Arguments
        N: Number of flashes.
        true_alpha: The true distance of the lighthouse along the shore.
        true_beta: The true distance of the lighthouse out to sea.
    
    Returns
        An array of alpha values, posterior probabilities for each x and the mean
        x position of flashes detected. 
    '''
    a = np.linspace(0., 5., 500)
    b = np.linspace(0., 5., len(a))
    log_prior = np.zeros((len(a), len(a)))
    alpha, beta = np.meshgrid(a, b)

    for n in range(N):
        flash = np.random.random()
        angle = flash * np.pi

        if angle > np.pi / 2:
            angle -= np.pi 

        x = true_beta * np.tan(angle) + true_alpha
        log_posterior = log_prior + np.log(beta) - np.log(cauchy(x, alpha, beta))
        log_prior = log_posterior

    if N != 0:
        log_prior -= log_prior.max()

    prior = np.exp(log_prior)
    
    return alpha, beta, prior

def plot_alpha(N, true_alpha, true_beta):
    '''
    Plots the results from sim_alpha (able to do so for a list of the number of 
    flashes.

    Arguments
        N: Number of flashes.
        true_alpha: The true distance of the lighthouse along the shore.
        true_beta: The true distance of the lighthouse out to sea.
    '''
    plt.suptitle('$P(\\alpha | \{x_k\}, I)$')

    for i, n in enumerate(N):
        alpha, res, mean = sim_alpha(n, true_alpha, true_beta)
        plt.axvline(mean, color='black')
        plt.subplot(4, 4, i+1)
        plt.plot(alpha, res)
        plt.title('n = {}'.format(n))
        plt.xlabel('$\\alpha$')
        plt.ylabel('$P(\\alpha | \{x_k\}, \\beta, I)$')
    
    plt.tight_layout()
    plt.show()

def plot_alpha_beta(N, true_alpha, true_beta):
    '''
    Plots the results from sim_alpha_beta (able to do so for a list of the number 
    of flashes.

    Arguments
        N: Number of flashes.
        true_alpha: The true distance of the lighthouse along the shore.
        true_beta: The true distance of the lighthouse out to sea.
    '''
    plt.suptitle('$P(\\alpha, \\beta | \{x_k\}, I)$')

    for i, n in enumerate(N):
        alpha, beta, res = sim_alpha_beta(n, true_alpha, true_beta)
        plt.subplot(4, 4, i+1)
        plt.contourf(alpha, beta, res)
        plt.title('n = {}'.format(n))
        plt.xlabel('$\\alpha$')
        plt.ylabel('$P(\\alpha | \{x_k\}, \\beta, I)$')
    
    plt.tight_layout()
    plt.show()

# run
N1 = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
N2 = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
true_alpha = 3.8
true_beta = 2.5

plot_alpha(N1, true_alpha, true_beta)
plot_alpha_beta(N2, true_alpha, true_beta)

'''
N = 4096
alpha, post, mean_x = sim(N, true_alpha, true_beta)
print(mean_x)
plt.plot(alpha, post)
plt.show()
'''
