import numpy as np
import matplotlib.pyplot as plt

def cauchy(x, alpha, beta):
    return np.power(beta, 2.) + np.power((x - alpha), 2.)

def sim_alpha(N, true_alpha, true_beta):
    alpha = np.linspace(0., 10., 500)
    log_prior = np.zeros(len(alpha))
    xk = np.array([])

    for n in range(N):
        flash = np.random.random()
        angle = flash * np.pi
        
        if angle > np.pi / 2:
            angle -= np.pi 

        x = true_beta * np.tan(angle) + true_alpha
        xk = np.append(xk, x)
        log_posterior = log_prior - np.log(cauchy(x, alpha, true_beta))
        '''
        likelihood = cauchy(x, alpha, true_beta)
        posterior = likelihood * prior
        prior = posterior
        '''
        log_prior = log_posterior
    
    if N != 0:
        log_prior -= log_prior.max()
    
    prior = np.exp(log_prior)
    return alpha, prior, xk.mean()

def sim_alpha_beta(N, true_alpha, true_beta):
    alpha = np.linspace(0., 5., 500)
    beta = np.linspace(0., 5., len(alpha))
    log_prior = np.zeros((len(alpha), len(alpha)))
    a, b = np.meshgrid(alpha, beta)
    print('LOG PRIOR\n')
    print(log_prior)
    print('ALPHA\n')
    print(a)
    print('BETA\n')
    print(b)
    xk = np.array([])

    for n in range(N):
        flash = np.random.random()
        angle = flash * np.pi

        if angle > np.pi / 2:
            angle -= np.pi 

        x = true_beta * np.tan(angle) + true_alpha
        xk = np.append(xk, x)
        log_posterior = log_prior - np.log(cauchy(x, a, b))
        log_prior = log_posterior

    if N != 0:
        log_prior -= log_prior.max()

    prior = np.exp(log_prior)
    print('X\n')
    print(xk)
    print('POST\n')
    print(prior)
    return alpha, beta, prior, xk.mean()

def plot_alpha(N, true_alpha, true_beta):
    plt.suptitle('$P(\\alpha | \{x_k\}, I)$')

    for i, n in enumerate(N):
        alpha, res, mean = sim_alpha(n, true_alpha, true_beta)
        plt.subplot(4, 4, i+1)
        plt.plot(alpha, res)
        plt.title('n = {}'.format(n))
        plt.xlabel('$\\alpha$')
        plt.ylabel('$P(\\alpha | \{x_k\}, \\beta, I)$')
    
    print('mean x: {}'.format(mean))
    plt.tight_layout()
    plt.show()

def plot_alpha_beta(N, true_alpha, true_beta):
    plt.suptitle('$P(\\alpha, \\beta | \{x_k\}, I)$')

    for i, n in enumerate(N):
        alpha, beta, res, mean = sim_alpha_beta(n, true_alpha, true_beta)
        plt.subplot(4, 4, i+1)
        plt.contourf(alpha, beta, res)
        plt.title('n = {}'.format(n))
        plt.xlabel('$\\alpha$')
        plt.ylabel('$P(\\alpha | \{x_k\}, \\beta, I)$')
    
    print('mean x: {}'.format(mean))
    plt.tight_layout()
    plt.show()

# run
N1 = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
N2 = [0, 1, 2, 4]
true_alpha = 1.
true_beta = 1.

plot_alpha(N1, true_alpha, true_beta)
plot_alpha_beta(N2, true_alpha, true_beta)
'''
N = 4096
alpha, post, mean_x = sim(N, true_alpha, true_beta)
print(mean_x)
plt.plot(alpha, post)
plt.show()
'''

