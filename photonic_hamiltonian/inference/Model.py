from pytensor import tensor as pt
import numpy as np
import pymc as pm
import pytensor

def Hamiltonian_model(data, priors):
    """
    returns a pymc model to infer the parameters for a 4-basis Hamiltonian.
    The piors on all parameters are either Uniform, Gaussian, or Exponential
    The likelihood ~ Gaussian(line, sigma_y)
    Line is the expectation value obtained by taking the sum of background and 4 Lorentzian peaked
    at the eigenvalues of 4x4 Hamiltonian matrix
    
    Parameters:
        data: the data set of a single-momentum spectrum 
              (DataArray with 'normf': frequency, 'spectrum': intensity, and 'spectrum_std': intensity uncertainty)
        priors: the prior types and prior coefficients of all parameters we want to infer
              (dict mapping a string, name of the parameter, to a tuple (prior_type, [prior coefficients]))
        For example: {'param 1':('Uniform', [lower bound, upper bound]),
                    'param 2':('Gaussian', [mean, std])
                    'param 3':('Exponential', [mean]}
              
    Return:
        ham_model: the pymc model can be used to infer the parameters and their posterior
              (pymc Model object)
    
    """
    #extract data to numpy arrays
    freq = data['normf'].to_numpy()
    intensity = data['spectrum'].to_numpy()
    intensity_sig = data['spectrum_std'].to_numpy()
    
    #define likelihood function
    def model(theta, x):
        """
        returns the loglike likelihood of our model
        
        Parameters:
            theta: the parameters in Hamiltonian matrix (list)
            y, x: our data (numpy arrays)
        
        Return:
            the model prediction (float)
        """
        #for our 4-basis Hamiltonian, the parameters include
        #interaction-between-modes terms u11, u20;
        #background A0 and heights of 4 peaks A1, A2, A3, A4
        #peak width (assumed to be the same for all peaks) sigma_L
        u11, u20, e0, de, A0, A1, A2, A3, A4, W1, W2, W3, W4 = theta
        
        #energy of each mode is assumed to be fixed
        ex = e0 + de   # for k = (0, +-0.05) the energy of uncoupled slab mode 1 
        ey = e0 - de   # for k = (+-0.05,0) the energy of uncoupled slab mode 2
        
        #Hamiltonian matrix
        ham_np = np.array([[ex,u11,u20,u11],
                           [u11,ey,u11,u20],
                           [u20,u11,ey,u11],
                           [u11,u20,u11,ex]])
        ham = pytensor.shared(np.zeros((4,4)))
        for row in range(4):
            for col in range(4):
                ham = pt.set_subtensor(ham[row, col], ham_np[row, col])
        
        #peak heights, peak positions, and peak widths
        An_np = np.array([A1,A2,A3,A4])
        Wn_np = np.array([W1,W2,W3,W4])
        An = pytensor.shared(np.zeros(4))
        Wn = pytensor.shared(np.zeros(4))
        for col in range(4):
            An = pt.set_subtensor(An[col], An_np[col])
            Wn = pt.set_subtensor(Wn[col], Wn_np[col])

        Cn = pt.nlinalg.eigh(ham)[0]
        # make sure eigenvalues are sorted
        Cn = pt.sort(Cn)
        
        #expectation value as sum of 4 gaussian peaks and background
        #VINNY EDITS: make thispart extensible. Probably use a loop.pytensor sort of thing, or make a new pytensor and do pt.sum
        line = A0
        for i in range(4):
            line = pt.sum(line, An[i] * pt.exp(-pt.sqr(x - Cn[i]) / (2 * pt.sqr(Wn[i]))))
        return line
    #create the multi Gaussian peak model
    ham_model = pm.Model()
    with ham_model:
        # Priors for unknown model parameters
        theta_list = []
        for i in list(priors.keys()):
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
    u11, u20, e0, de, A0, A1, A2, A3, A4, W1, W2, W3, W4 = theta
    ex = e0 + de   # for k = (0, +-0.05) the energy of uncoupled slab mode 1 
    ey = e0 - de    # for k = (+-0.05,0) the energy of uncoupled slab mode 2
    An = [A1, A2, A3, A4]
    Wn = [W1, W2, W3, W4]
    
    #Hamiltonian matrix and its eigenvalues as line peaks
    H = [[ex,u11,u20,u11],
         [u11,ey,u11,u20],
         [u20,u11,ey,u11],
         [u11,u20,u11,ex]]
    Cn = np.real(np.linalg.eigvals(H))
    Cn = np.sort(Cn)
    
    #calculate normalized intensity
    line_each = [Ai * np.exp(-(freq - Ci)**2 / (2 * Wi**2)) for Ai, Ci, Wi in zip(An, Cn, Wn)]
    line = np.sum(line_each, axis=0) + A0
    
    return line