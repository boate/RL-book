import matplotlib.pyplot as plt
import numpy as np

def optimal_val(
    alpha: float,
    u: float, 
    r: float, 
    sigma: float):

    numerator = -(u - r) - 2.0*(1000000.0 + r)*(u-r)
    denom = -alpha*sigma*sigma + 2.0*(u-r)*(u-r)

    return numerator/denom

if __name__ == '__main__':
    alphas = np.linspace(1, 100)
    u = 2.0
    r = 1.0
    sigma = 5.0
    optimals = []
    for alpha in alphas:
        optimals.append(optimal_val(alpha, u, r, sigma))
    
    plt.plot(alphas, optimals)
    plt.show()
