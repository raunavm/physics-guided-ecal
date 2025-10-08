# physics-guided-ecal
A Julia program for a physics guided electromagnetic calorimeter (ECAL) layer model. Implements a graph / finite-difference Laplacian + small neural residual (Lux) and trains with ADAM with NN first and then L-BFGS fine-tune with physics enabled. Includes data loaders (HDF5/XML), training, evaluation and plotting utilities.
