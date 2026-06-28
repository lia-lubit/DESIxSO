COBAYA PROJECT TODO LIST

CHAIN 0: add primary CMB spectra to MCMC and Fisher
-- modify plotting function to plot CMB primary spectra -- Done
-- figure out how to modify likelihood with primary spectra
-- figure out how to modify covariance with primary spectra
-- figure out how to update data vector with primary spectra (including noise)
-- modify all spectra, noise, vector, and matrix functions to include primary spectra, if desired
-- update Fisher to include primary spectra
-- run Fisher w/ omega_m, A_s and omega_m using primary CMB 
-- update Cobaya to include primary spectra
-- run MCMC w/ omega_m, A_s and omega_m using primary CMB

CHAIN 1: test setup and restuls against published forecasts for configurations
-- mimic lens and source bins and noise from Pratt -- done
-- run Fisher forecast and compare to Pratt -- done, results are off
-- test covariance matrices etc
-- figure out why the degeneracies are different

                                                   
---------- DONT PROCEED UNTIL I KNOW MY SET UP IS FULLY CORRECT -----------
                                                            
CHAIN 2: Compare constraints with photometric vs spectroscopic bins
IMPORTANT: run this only with lensing ratios -- e.g. use only k_c - g and k_g - g spectra
Run 1: 
- photometric bins
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




    
Others
would lensing ratios be sensitive to depth to reionization?
3. figure out what we expect given degeneracy
4. figure out how exactly this is using the lensing ratios...
5. figure out how to optimize


-- overlay constraints from DESI or CMB or other stuff to see how we match
-- see if direction in which we intersect is useful


MISC
-- just slice one photometric bin into however many bins it would be if it were spectroscopic
-- sanity check: slice what we already have
-- next step: match spectroscopic bins to one photometric bin
-- in theory lensing ratio only insensitive if we have the same scale cuts, but we dont have the same scale cuts, so well have some dependence on matter 
-- cmb lensing affect down to l of 5000?? -- just use l max at 5000 basically
-- see recent paper by Taylor Hoyt: https://arxiv.org/abs/2601.19424
-- figure out why the matrices are different with the plot function vs the normal method

Other
visualize spectra for diff omega_m? not sure what this was about 
marco bonicci
if pyccl is still slow after emulating the matter power spectrum, consider implementing the integrals ourselves 

NOTE: I got rid of the sorting line in create_simplified_desired_pairs because I realized it didnt match the normal f_map -- I dont think this will cause problems, but in case it does, Im leaving a note -- June 10, 2026, 16:30