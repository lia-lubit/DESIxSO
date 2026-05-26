COBAYA PROJECT TODO LIST
1. Confirm P(k) emulator is working
2. get from P(k) emulator to relevant spectra
3. test time diff and accuracy diff from PyCCL
4. build this into Cobaya -- new likelihood?
5. spawn more chains and use NERSC exclusive node
6. rerun omega_m and check results
7. run for omega_m and sigma_8 --> if lensing ratios are working as we think, we should not be constraining sigma8 at all (or a_s)

Later
1. Make trace and corner plot functions in functions_and_classes.py to make this clean
2. Make covariance and data saving function?

Other
change to uniform prior?
marco bonicci
if pyccl is still slow after emulating the matter power spectrum, consider implementing the integrals ourselves 
once we do this, we can start exploring different configurations and start seeing which ones are most interesting and start analyzing data