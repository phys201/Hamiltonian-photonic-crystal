from pytensor import tensor as pt
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
    freq = data['x'].to_numpy()
    intensity = data['y'].to_numpy()
    intensity_sig = data['sigma_y'].to_numpy()
    
    def likelihood(theta, y, x, sigma_y):
        """
        returns the loglike likelihood of our model
        
        Parameters:
            theta: the parameters in Hamiltonian matrix (list)
            y, x, sigma_y: our data (numpy arrays)
        
        Return:
            the loglike likelihood (float)
        """
        #for our four-basis Hamiltonian, the parameters include
        #interaction-between-modes terms h11, u20;
        #background A0 and heights of 4 peaks A1, A2, A3, A4
        #peak width (assumed to be the same for all peaks) sigma_L
        h11, u20, A0, A1, A2, A3, A4, sigma_L = theta
        
        #energy of each mode is assumed to be fixed
        C = 0.6346    # for k = (0.05,0) the eigen frequency of uncoupled slab mode 1 (*2 degeneracy)
        C2 = 0.669    # for k = (0.05,0) the eigen frequency of uncoupled slab mode 2 (*2 degeneracy)
        
        #Hamiltonian matrix
        ham = [[C2,h11,u20,h11],
               [h11,C,h11,u20],
               [u20,h11,C,h11],
               [h11,u20,h11,C2]]
        
        #peak heights and peak positions
        An = [A1,A2,A3,A4]
        Cn = np.real(np.linalg.eigvals(ham))
        
        #expectation value as sum of 4 gaussian peaks and background
        gaussian_list = [Ai * np.exp(-(x - Ci)**2 / (2 * sigma_L**2)) for Ai, Ci in zip(An, Cn)]
        line = A0 + np.sum(gaussian_list,axis=0)
        
        return np.sum(-(0.5 / sigma_y**2) * (y - line) ** 2)
        
    class LogLike(pt.Op):
        """
        Passing the Op a vector of values (the parameters that define our model)
        and returning a scalar value of loglike likelihood.
        """
        itypes = [pt.dvector]  # expects a vector of parameter values when called
        otypes = [pt.dscalar]  # outputs a vector of peak positions

        def __init__(self, likelihood, y, x, sigma_y):
            """
            Initialise the Op with things that our log-likelihood function
            requires.
    
            Parameters
            ----------
            likelihood:
                The log-likelihood function we've defined
            y, x, sigma_y:
                Our data
            """

            # add inputs as class attributes
            self.likelihood = likelihood #your Hamiltonian function goes in here
            self.x = x
            self.y = y 
            self.sigma = sigma_y

        def perform(self, node, inputs, outputs):
            # the method that is used when calling the Op
            (theta,) = inputs  # this will contain all variables

            # call the loglike function
            model = self.likelihood(theta, self.y, self.x, self.sigma)
            
            outputs[0][0] = np.array(model)  # output the log-likelihood
    
    ham_model = pm.Model()
    logl = LogLike(likelihood, intensity, freq, intensity_sig)
    with ham_model:
        # Priors for unknown model parameters
        theta_list = []
        for i in list(prior_bounds.keys()):
            theta_list.append(pm.Uniform(i, lower=prior_bounds[i][0], upper=prior_bounds[i][1]))
    
        #input of our log-likelihood
        theta = pt.as_tensor_variable(theta_list)
    
        # Likelihood of observations
        pm.Potential("likelihood", logl(theta))

    return ham_model