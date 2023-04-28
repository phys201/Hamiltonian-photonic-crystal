from pytensor import tensor as pt
import numpy as np
import pymc as pm
import pytensor

def Hamiltonian_model(data, prior_bounds):
    """
    returns a pymc model to infer the parameters for a four-basis Hamiltonian.
    The piors on all parameters ~ Uniform(given lower, given upper)
    The likelihood ~ Gaussian(line, sigma_y)
    Line is the expectation value obtained by taking the sum of background and 4 Gaussian peaked
    at the eigenvalues of 4x4 Hamiltonian matrix
    
    Parameters:
        data: the data set of a single-momentum spectrum 
              (DataArray with 'x': frequency, 'y': intensity, and 'sigma_y': intensity uncertainty)
        prior_bounds: the bounds for uniform priors of all parameters we want to infer
              (dict mapping a string, name of the parameter, to a list of its bounds [lower, upper])
              
    Return:
        ham_model: the pymc model can be used to infer the parameters and their posterior
              (pymc Model object)
    
    """
    #extract data to numpy arrays
    freq = data['normf'].to_numpy()
    intensity = data['spectrum'].to_numpy()
    intensity_sig = data['spectrum_std'].to_numpy()
    
    #define likelihood function
    def model(theta, y, x):
        """
        returns the loglike likelihood of our model
        
        Parameters:
            theta: the parameters in Hamiltonian matrix (list)
            y, x, sigma_y: our data (numpy arrays)
        
        Return:
            the loglike likelihood (float)
        """
        #for our four-basis Hamiltonian, the parameters include
        #interaction-between-modes terms u11, u20;
        #background A0 and heights of 4 peaks A1, A2, A3, A4
        #peak width (assumed to be the same for all peaks) sigma_L
        u11, u20, A0, A1, A2, A3, A4, sigma_L = theta
        
        #energy of each mode is assumed to be fixed
        ex = 0.669   # for k = (0, +-0.05) the energy of uncoupled slab mode 1 
        ey = 0.6346    # for k = (+-0.05,0) the energy of uncoupled slab mode 2
        
        #Hamiltonian matrix
        ham_np = np.array([[ex,u11,u20,u11],
                           [u11,ey,u11,u20],
                           [u20,u11,ey,u11],
                           [u11,u20,u11,ex]])
        ham = pytensor.shared(np.zeros((4,4)))
        for row in range(4):
            for col in range(4):
                ham = pt.set_subtensor(ham[row, col], ham_np[row, col])
        
        #edit the pytensors so it's cleaner: how to write some constants and some are pytensors
        #peak heights and peak positions
        An_np = np.array([A1,A2,A3,A4])
        An = pytensor.shared(np.zeros(4))
        for col in range(4):
            An = pt.set_subtensor(An[col], An_np[col])

        Cn = pt.nlinalg.eigh(ham)[0]
        # make sure eigenvalues are sorted
        Cn = pt.sort(Cn)
        
        #expectation value as sum of 4 gaussian peaks and background
        #VINNY EDITS: make thispart extensible. Probably use a loop.pytensor sort of thing, or make a new pytensor and do pt.sum
        return A0 + An[0] * pt.exp(-pt.sqr(x - Cn[0]) / (2 * pt.sqr(sigma_L))) + An[1] * pt.exp(-pt.sqr(x - Cn[1]) / (2 * pt.sqr(sigma_L))) + An[2] * pt.exp(-pt.sqr(x - Cn[2]) / (2 * pt.sqr(sigma_L))) + An[3] * pt.exp(-pt.sqr(x - Cn[3]) / (2 * pt.sqr(sigma_L)))
    #create the multi Gaussian peak model
    ham_model = pm.Model()
    with ham_model:
        # Priors for unknown model parameters
        theta_list = []
        for i in list(prior_bounds.keys()):
            #check if the lower bound is strictly smaller than the upper bound
            if prior_bounds[i][0] >= prior_bounds[i][1]:
                raise ValueError
            theta_list.append(pm.Uniform(i, lower=prior_bounds[i][0], upper=prior_bounds[i][1]))
    
        #input of our log-likelihood
        theta = pt.as_tensor_variable(theta_list)
        model_predictions = model(theta,intensity,freq)
        # Likelihood of observations
        likelihood = pm.Normal('likelihood',mu = model_predictions, sigma = intensity_sig, observed = intensity)
    return ham_model

def fit_curve(freq, theta):
    """
    Calculate the multi Gaussian peak curve fit using a set of given fitting parameters

    Parameters
    ----------
    freq (NumPy array):
        The array of normalized frequency (x data)
    theta (list):
        The list of fitting parameters, in the order u11, u20, A0, A1, A2, A3, A4, sigma_L
        
    Return
    ---------
    line (NumPy array):
        The array of corresponding fitted intensity
    """
    ex = 0.669   # for k = (0, +-0.05) the energy of uncoupled slab mode 1 
    ey = 0.6346    # for k = (+-0.05,0) the energy of uncoupled slab mode 2
    u11, u20, A0, A1, A2, A3, A4, sigma_L = theta
    An = [A1, A2, A3, A4]
    
    #Hamiltonian matrix and its eigenvalues as line peaks
    H = [[ex,u11,u20,u11],
         [u11,ey,u11,u20],
         [u20,u11,ey,u11],
         [u11,u20,u11,ex]]
    Cn = np.real(np.linalg.eigvals(H))
    Cn = np.sort(Cn)
    
    #calculate normalized intensity
    line_each = [Ai * np.exp(-(freq - Ci)**2 / (2 * sigma_L**2)) for Ai, Ci in zip(An, Cn)]
    line = np.sum(line_each, axis=0) + A0
    
    return line