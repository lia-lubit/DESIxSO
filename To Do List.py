COBAYA PROJECT TODO LIST
1. how much do we gain if we slice into smaller bins? eg spectroscopic instead of photometric -- ONLY WITH k_c - g and k_g - g g is lens galaxies 
-- just slice one photometric bin into however many bins it would be if it were spectroscopic
-- sanity check: slice what we already have
-- next step: match spectroscopic bins to one photometric bin
-- is the noise the same? nope youll have to adjust the shot noise and shape noise 
-- in theory lensing ratio only insensitive if we have the same scale cuts, but we dont have the same scale cuts, so well have some dependence on matter 
-- cmb lensing affect down to l of 5000?? -- just use l max at 5000 basically
-- compare for w0 and wa and omega_m with DESI spectroscopic vs DESI photometric
-- first test: fix everything to planck values except w0 wa and omega_m -- use broad uniform priors for the varying ones
-- second test: put Planck priors on all cosmological priors except w0 wa and omega_m
-- fiducial data at w0wa that best fits DESI data
-- see recent paper by Taylor Hoyt
-- https://arxiv.org/abs/2601.19424

2. test setup and restuls against published forecasts for configurations
3. learn about fisher matrices -- already have the covariance matrices 
"im sure you could ace that in a week"
-- could allow us to more efficiently look at configuration
explore wide range of configurations with fisher
full blown mcmc for interesting ones
full blown mcmc + custom emulator? for most interesting ones


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
