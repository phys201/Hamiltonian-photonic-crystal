from pytensor import tensor as pt
import numpy as np
import pymc as pm
import pytensor

def prediction_model(theta, x):
    """
    Calculate the multi Lorentzian peak curve fit using a set of given fitting parameters

    Parameters
    ----------
    x (NumPy array):
        The array of normalized frequency
    theta (list of Floats or PyTensors):
        The list of fitting parameters, in the order u11, u20, e0, de, A0, A1, A2, A3, A4, Q1, Q2, Q3, Q4
        
    Return
    ---------
    line (NumPy array or PyTensors):
        The array of corresponding fitted intensity
    """
    #for our 4-basis Hamiltonian, the parameters include
    #energy and its deviation of uncoupled modes in uniform slab: e0, de
    #interaction-between-modes terms u11, u20;
    #background intensity A0
    #heights of 4 peaks A1, A2, A3, A4
    #quality factors of 4 peaks Q1, Q2, Q3, Q4
    u11, u20, e0, de, A0, A1, A2, A3, A4, Q1, Q2, Q3, Q4 = theta

    #energy of uncoupled modes
    ex = e0 + de   # relatively high energy uncoupled modes. 2-fold degeneracy for C4 lattice along GM direction in H matrix
    ey = e0 - de   # relatively low energy uncoupled modes. 2-fold degeneracy for C4 lattice along GM direction in H matrix
    
    #Hamiltonian matrix
    ham_np = np.array([[ex,u11,u20,u11],
                 [u11,ey,u11,u20],
                 [u20,u11,ey,u11],
                 [u11,u20,u11,ex]])
    An_np = np.array([A1,A2,A3,A4])
    Qn_np = np.array([Q1,Q2,Q3,Q4])
    
    #Diagonalize the matrix in two cases
    #if diagonalize a Numpy array matrix object
    if isinstance(theta[0], pt.TensorVariable) == False:
        Cn_np = np.real(np.linalg.eigvals(ham_np))    # center frequency of each peak
        Cn_np = np.sort(Cn_np)
        Wn_np = Cn_np/Qn_np                    # width (FWHM) of each peak
        #calculate normalized intensity
        line_each = [(Ai * Wi**2) / ((x- Ci)**2 + Wi**2) for Ai, Ci, Wi in zip(An_np, Cn_np, Wn_np)]
        line = np.sum(line_each, axis=0) + A0
    #if diagonalize a PyTensor matrix object
    else:
        ham = pytensor.shared(np.zeros((4,4)))
        for row in range(4):
            for col in range(4):
                ham = pt.set_subtensor(ham[row, col], ham_np[row, col])

        #peak heights, peak positions, and peak widths
        An = pytensor.shared(np.zeros(4))
        Qn = pytensor.shared(np.zeros(4))
        for col in range(4):
            An = pt.set_subtensor(An[col], An_np[col])
            Qn = pt.set_subtensor(Qn[col], Qn_np[col])

        Cn = pt.nlinalg.eigh(ham)[0]          # center frequency of each peak
        Cn = pt.sort(Cn)
        Wn = Cn/Qn                      # width (FWHM) of each peak
        line = A0 + An[0] * pt.sqr(Wn[0]) / (pt.sqr(x-Cn[0]) + pt.sqr(Wn[0])) + An[1] * pt.sqr(Wn[1]) / (pt.sqr(x-Cn[1]) + pt.sqr(Wn[1])) + An[2] * pt.sqr(Wn[2]) / (pt.sqr(x-Cn[2]) + pt.sqr(Wn[2])) + An[3] * pt.sqr(Wn[3]) / (pt.sqr(x-Cn[3]) + pt.sqr(Wn[3]))
    
    return line

def Hamiltonian_model(data, priors):
    """
    returns a pymc model to infer the parameters for a 4-basis Hamiltonian.
    The piors on all parameters are either Uniform, Gaussian, or Exponential
    The likelihood ~ Gaussian(line, sigma_y)
    Line is the expectation value obtained by taking the sum of background and 4 Lorentzians peaked
    at the eigenvalues of 4x4 Hamiltonian matrix
    
    Parameters:
        data: the data set of a single-momentum spectrum 
              (DataArray with 'normf': frequency, 'y1', 'y2', 'y3', 'y4' are results of 4 repeated measurements)
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
    namef = ['y1', 'y2', 'y3', 'y4']      #measurements are repeated 4 times
    # intensity = np.zeros((4,Nf))
    intensity = np.array([data[i].to_numpy() for i in namef])
    
    #create the multi Lorentzian peak model
    ham_model = pm.Model()
    with ham_model:
        # Priors for unknown model parameters
        theta_list = []
        for i in list(priors.keys()):
            if priors[i][0] == 'Uniform':
                lower, upper = priors[i][1]
                theta_list.append(pm.Uniform(i, lower, upper))
            elif priors[i][0] == 'Normal':
                mean, std = priors[i][1]
                theta_list.append(pm.Normal(i, mean, std))
            elif priors[i][0] == 'Exponential':
                mean, = priors[i][1]
                theta_list.append(pm.Exponential(i, mean))
            else:
                raise ValueError('Invalid prior type')
    
        #input of the model prediction
        theta = pt.as_tensor_variable(theta_list[:-1])
        line = prediction_model(theta, freq)
        
        # Uncertainty of the intensity is proportional to square root of intensity, sigma_y represent the ratio
        sigma_y = theta_list[-1]
        
        # Gaussian Likelihood of observations
        likelihood = pm.Normal('likelihood', mu = line, sigma = sigma_y*pt.sqrt(intensity), observed = intensity)
    return ham_model