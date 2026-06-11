COBAYA PROJECT TODO LIST

MISC
-- just slice one photometric bin into however many bins it would be if it were spectroscopic
-- sanity check: slice what we already have
-- next step: match spectroscopic bins to one photometric bin

-- in theory lensing ratio only insensitive if we have the same scale cuts, but we dont have the same scale cuts, so well have some dependence on matter 
-- cmb lensing affect down to l of 5000?? -- just use l max at 5000 basically

-- see recent paper by Taylor Hoyt: https://arxiv.org/abs/2601.19424
-- figure out why the matrices are different with the plot function vs the normal method

CHAIN 1: Compare constraints with photometric vs spectroscopic bins
IMPORTANT: run this only with lensing ratios -- e.g. use only k_c - g and k_g - g spectra
Run 1: 
- one photometric bin
- set fiducial cosmology to w0wa cosmology that best fits DESI data
- let n_ell ~ 5000
- fix everything to Planck values except w0, wa, and omega_m
- use broad uniform priors for w0, wa, and omega_m
Run 2: 
- spectroscopic bin that overlap that photometric bin
- use new shot and shape noise 
- set fiducial cosmology to w0wa cosmology that best fits DESI data
- let n_ell ~ 5000
- fix everything to Planck values except w0, wa, and omega_m
- use broad uniform priors for w0, wa, and omega_m
Compare: timing and constraining power
Run 3: 
- one photometric bin
- set fiducial cosmology to w0wa cosmology that best fits DESI data
- let n_ell ~ 5000
- put uniform planck priors of all cosmological parameters except w0, wa, and omega_m
- use broad uniform priors for w0, wa, and omega_m
Run 4: 
- spectroscopic bin that overlap that photometric bin
- use new shot and shape noise 
- set fiducial cosmology to w0wa cosmology that best fits DESI data
- let n_ell ~ 5000
- put uniform planck priors of all cosmological parameters except w0, wa, and omega_m
- use broad uniform priors for w0, wa, and omega_m
Compare: timing and constraining power


    
CHAIN 2: test setup and restuls against published forecasts for configurations

CHAIN 3: learn about fisher matrices -- already have the covariance matrices 
-- could allow us to more efficiently look at configuration
explore wide range of configurations with fisher
full blown mcmc for interesting ones
full blown mcmc + custom emulator? for most interesting ones




Others
would lensing ratios be sensitive to depth to reionization?
2. run for omega_m and sigma_8 with slight variation in starting points (right now its running with them all starting at the same time)
3. figure out what we expect given degeneracy
4. figure out how exactly this is using the lensing ratios...
5. figure out how to optimize


-- overlay constraints from DESI or CMB or other stuff to see how we match
-- see if direction in which we intersect is useful

Other
visualize spectra for diff omega_m? not sure what this was about 
marco bonicci
if pyccl is still slow after emulating the matter power spectrum, consider implementing the integrals ourselves 
once we do this, we can start exploring different configurations and start seeing which ones are most interesting and start analyzing data
1. Make trace and corner plot functions in functions_and_classes.py to make this clean
2. Make covariance and data saving function?



NOTE: I got rid of the sorting line in create_simplified_desired_pairs because I realized it didnt match the normal f_map -- I dont think this will cause problems, but in case it does, Im leaving a note -- June 10, 2026, 16:30