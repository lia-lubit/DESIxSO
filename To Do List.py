COBAYA PROJECT TODO LIST
0. get chi 2 for diff omega_m
    minimizer?

when you use the emulator,make it self consistent -- use it in the covariance and data vector too

Later
0. run for omega_m and sigma_8 --> if lensing ratios are working as we think, we should not be constraining sigma8 at all (or a_s)
1. Make trace and corner plot functions in functions_and_classes.py to make this clean
2. Make covariance and data saving function?

Other
change to uniform prior?
marco bonicci
if pyccl is still slow after emulating the matter power spectrum, consider implementing the integrals ourselves 
once we do this, we can start exploring different configurations and start seeing which ones are most interesting and start analyzing data