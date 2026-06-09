COBAYA PROJECT TODO LIST
1. run for A_s and omega_m with just k_c x g and k_g x g in data vector/cov matrix (cmb lensing -- galaxy density, galaxy-lensing, galaxy-density)
2. is the matrix and vector different if only ever consider a subset? or will the subset of the vector that you keep be identical to the subset of the full one? -- usually eg the covariance matrix for k_c x g and k_g x g will have g x g terms, etc. -- what becomes of those?
3. why is theplotting not working?
    
2. run for omega_m and sigma_8 with slight variation in starting points (right now its running with them all starting at the same time)
3. figure out what we expect given degeneracy
4. figure out how exactly this is using the lensing ratios...
5. figure out how to optimize

Other
visualize spectra for diff omega_m? not sure what this was about 
marco bonicci
if pyccl is still slow after emulating the matter power spectrum, consider implementing the integrals ourselves 
once we do this, we can start exploring different configurations and start seeing which ones are most interesting and start analyzing data
1. Make trace and corner plot functions in functions_and_classes.py to make this clean
2. Make covariance and data saving function?
