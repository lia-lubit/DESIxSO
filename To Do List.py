COBAYA PROJECT TODO LIST

1. Create project folder structure
   - utils.py
   - so_x_desi_likelihood.py
   - likelihood.yaml
   - data/ folder

2. Move helper functions into utils.py
   - ForecastMap
   - build_covariance_from_data
   - build_tracers_from_data
   - build_tracer_dict
   - build_noise_dict
   - build_spectra_dict
   - any other helper / physics functions

3. Remove notebook dependencies
   - Remove eval(...)
   - Remove all global variables (DESI_lens_data, etc.)
   - Ensure everything comes from imports or file paths

4. Save all data to disk (.npy files)
   - lens_data → data/lens.npy
   - source_data → data/source.npy
   - covariance_matrix → data/covariance.npy
   - observed_data_vector → data/data_vector.npy

5. Load data inside likelihood class (NOT notebook)
   - Use np.load(self.data_specs["..._path"])

6. Create so_x_desi_likelihood.py
   - Import Likelihood, numpy, ccl
   - Import everything from utils.py
   - Define SO_x_DESI_Likelihood class only

7. Create likelihood.yaml
   - Define python_class: SO_x_DESI_Likelihood
   - Set python_path: .
   - Provide data_specs (file paths + constants)
   - Define sampler settings

8. Test imports BEFORE running Cobaya
   - from so_x_desi_likelihood import SO_x_DESI_Likelihood
   - import utils

9. Run Cobaya
   - cobaya-run likelihood.yaml

10. Debug if needed
   - Check file paths
   - Check module imports
   - Check array shapes (data vector vs covariance)