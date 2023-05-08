# Hamiltonian Band Structure for Photonic Crystals

This package can be used to study how to properly choose the basis for the Hamiltonian model of photonic crystals band structures

## Background
Photonic crystals (PhC) are micro/nano artificial atoms composed of periodically perturbed refractive indices dielectric material, which can manipulate the propagation of electromagnetic (EM) waves in a similar manner to how actual crystals manipulate electron waves. The periodic symmetry and Maxwell equations guarantee that the eigenmodes of  EM waves can be held in PhC are represented by photonic band structures. One popular method to analyze the band structures of photonic crystals is to apply the Hamiltonian formalism, however, the method to extract coupling parameters of Hamiltonian matrix from experiment data has been left unrevealed in the majority of related papers. This will be the main purpose of our project

## Data format
The raw data we will use is the transmitted spectrum with momentum resolution of a square-lattice (C4 symmetry) photonic crystal slab, measured by a CCD camera. The data set can be represented as $I(i,j,f)$ , where I refers to brightness of a CCD pixel (value between 0 and 255 in grayscale); $(i,j)$ is the label of pixels (320x240), corresponding to $(k_x, k_y)$ - the momentum of the optical response; $f$ refers to the frequency of input laser (wavelength between 1100 nm and 1700 nm). 

## Models
The Hamiltonian matrix  consists of coupling parameters that describe the interaction between different slab modes in the basis. At each value of $(k_x, k_y)$, the energy levels can be obtained by diagonalizing the Hamiltonian matrix and finding its eigenvalues. Our model to fit the single-$k$ spectrum is a multiple-Lorentzian-peak curve with each peak centered at one energy level.

The number of bases corresponds to the dimension of the Hamiltonian matrix. Due to the C4 symmetry of our square lattice, the allowed basis sizes are 5, 9, 13,... We will perform parameter estimation for the 5 basis case.


## Technical approach
This package employs pymc3 for Markov Chain Monte Carlo (MCMC) sampling of generative models which represent the momentum-resolved transmitted intensity spectrum data upon shining laser light on a square-lattice (C4 symmetry) photonic crystal slab.
