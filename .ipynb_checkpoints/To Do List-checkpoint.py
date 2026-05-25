COBAYA PROJECT TODO LIST

1. Make trace and corner plot functions in functions_and_classes.py to make this clean
2. Make covariance and data saving function?

Figure out what is making the calculation take a long time
1. probably its making the power spectra
2. have more chains -- order 10 (also good to figure out how to make multiple chains and use different cores)
3. emulators exist which make this faster -- consider using this
4. we think that were insensitive to the power spectra, we could not recalculate the power spectra, assuming that it'll fall out
-- this is sketchy and eventually will stop working
5. cosmopower
6. marco bonicci

we think were only sensitive to sigma8
we should see sigma8 be very unconstrained and the omegam be constrained more tightly

change prior to uniform prior from Gaussian


1. spawn more chains
2. use emulator to make power spectrum calculators faster -- cosmopower emulator (read paper and use emulator)
3. change to uniform prior
4. run for omega_m and sigma_8 --> if lensing ratios are working as we think, we should not be constraining sigma8 at all (or a_s)

once we do this, we can start exploring different configurations and start seeing which ones are most interesting and start analyzing data