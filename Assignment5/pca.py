import numpy as np
import matplotlib.pyplot as plt

def pca(x):
    for i, sample in enumerate(x):
        sample -= np.mean(sample)
        x[i] = sample
    
    # Compute covariance
    x = x.T
    cov = np.cov(x)
    
    # Find eigenvalues, eigenvectors of covariance matrix
    e_val, e_vec = np.linalg.eig(cov)
    
    # Sort eigenvectors from largest eigenvalue to smallest
    sort_ind = np.argsort(np.absolute(e_val))[::-1]
    e_vec = e_vec[sort_ind].conjugate().T
    e_val = np.absolute(e_val)[sort_ind]
    
    # Compute projection of data onto new basis
    proj = np.dot(e_vec, x).T
    
    return e_val, e_vec, proj

# Simulate simple data
y = np.linspace(0., 50., 100)
noise_x = np.random.randint(low=0, high=50, size=np.size(y))
noise_y = np.random.randint(low=0, high=50, size=np.size(y))
x = 5 * y + noise_x 
sample1 = np.array([x, y]).T

# Simulate spring
noise_x1 = np.random.randint(low=0, high=50, size=np.size(y))
noise_x2 = np.random.randint(low=0, high=50, size=np.size(y))
noise_x3 = np.random.randint(low=0, high=50, size=np.size(y))
noise_y1 = np.random.randint(low=0, high=50, size=np.size(y))
noise_y2 = np.random.randint(low=0, high=50, size=np.size(y))
noise_y3 = np.random.randint(low=0, high=50, size=np.size(y))

y1 = y + noise_y1
y2 = y + noise_y2
y3 = y + noise_y3

x1 = 3 * y1 + noise_x1
x2 = 1 * y1 + noise_x2
x3 = -1.5 * y1 + noise_x3

sample2 = np.array([x1, y1, x2, y2, x3, y3]).T

# Simple data
e_val1, e_vec1, proj1 = pca(sample1)
print('Simple Data')
print('PRINCIPLE COMPONENTS')
print(e_vec1)

# Spring
e_val2, e_vec2, proj2 = pca(sample2)
print('\nSpring Data')
print('PRINCIPLE COMPONENTS')
print(e_vec2)

# Plot simple data
plt.figure(1)
plt.scatter(x, y, color='black')
plt.title('Simple Data')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('plots/simple_data.pdf')

# Plot spring data
plt.figure(2)
plt.scatter(x1, y1, color='blue')
plt.scatter(x2, y2, color='black')
plt.scatter(x3, y3, color='red')
plt.title('Spring Data')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('plots/spring_data.pdf')
plt.show()

