COBAYA PROJECT TODO LIST
1. Confirm P(k) emulator is working
2. get from P(k) emulator to relevant spectra
3. test time diff and accuracy diff from PyCCL

QUESTION: Should I be using weaklensingtraer for all my galaxy source bins?

1. Modify all standing functions to be workable with or without emulator -- DONE
2. Change calculate and plot functions to be workable with or without emulators
3. Change all cosmologies and functions to use A_s instead of sigma8 -- this will involve changing functions that use sigma8 to be ok with A_s 
-- might be worth having them depend on whether an emulator is being used or not
4. test covariance matrices with and without emulator
5. modify classes to be workable with or without emulator
6. make new likelihood that works with the emulator -- this will need the new cosmology
7. test omega_m with Cobaya
8. spawn more chains and use NERSC exclusive node
9. rerun omega_m and check results
10. run for omega_m and sigma_8 --> if lensing ratios are working as we think, we should not be constraining sigma8 at all (or a_s)

Later
1. Make trace and corner plot functions in functions_and_classes.py to make this clean
2. Make covariance and data saving function?

Other
change to uniform prior?
marco bonicci
if pyccl is still slow after emulating the matter power spectrum, consider implementing the integrals ourselves 
once we do this, we can start exploring different configurations and start seeing which ones are most interesting and start analyzing data