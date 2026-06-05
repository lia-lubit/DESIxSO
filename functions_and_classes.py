### FUNCTIONS AND CLASSES

# import tools

# standard library
import os
import sys
import inspect
import yaml
import time
import types

# scientific stack
import numpy as np
import jax.numpy as jnp
import scipy
from scipy.interpolate import interp1d
from scipy.stats import linregress
from scipy.integrate import simpson

# cosmology
import astropy
import astropy.units as units
from astropy.cosmology import LambdaCDM, w0waCDM
import pyccl as ccl
import camb
from camb import model, initialpower
import classy
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ

# plotting
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# inference / sampling
import cobaya
from cobaya.run import run as cobaya_run
from cobaya.likelihood import Likelihood
#from cobaya_utilities import tools -- this wont import, but I don't use it at the momment anyway
import getdist
from getdist import plots, MCSamples
from getdist.mcsamples import loadMCSamples

print("LOADING FILE:", os.path.abspath(__file__))

# -------------------------------------------------------------------------------------------------------------------------------------------- #

## FUNCTIONS

# calcualte and (maybe) plot angular power spectra
def calculate_and_plot_Cls(
    cosmology,        # ccl.Cosmology object
    source_data,         # 2D array: z, n_z_bin1, n_z_bin2, ...
    lens_data,          # 2D array: z, n_z_bin1, n_z_bin2, ...
    correlation_types=['GG', 'LL', 'GL', 'CC', 'CL', 'CG'], # List of correlation types to plot
    Pk2D_object = None,    # Pk2D object from emulator,
    plot = 'yes'       # chose whether to plot
):

    # extract lens and source redshift grids and distributions
    z_lens_grid = lens_data[:, 0]
    n_lens_dists = [lens_data[:, i] for i in range(1, lens_data.shape[1])]
    num_lens_bins = len(n_lens_dists)
    z_source_grid = source_data[:, 0]
    n_source_dists = [source_data[:, i] for i in range(1, source_data.shape[1])]
    num_source_bins = len(n_source_dists)
    z_CMB = 1090

    # initialize ccl.NumberCountsTracer for each lens and source bin
    lens_tracers_nc = []
    lensing_tracers_nc = []
    for i, n_z_lens_bin in enumerate(n_lens_dists):
        # use a constant linear bias of 1
        bias_values = np.ones_like(z_lens_grid)
        tracer = ccl.NumberCountsTracer(cosmology, has_rsd=False,
                                        dndz=(z_lens_grid, n_z_lens_bin),
                                        bias=(z_lens_grid, bias_values))
        lens_tracers_nc.append(tracer)
    for i, n_z_source_bin in enumerate(n_source_dists):
            # use a constant linear bias of 1
            tracer = ccl.WeakLensingTracer(cosmology, dndz=(z_source_grid, n_z_source_bin))
            lensing_tracers_nc.append(tracer)

    # create CMB lensing tracerrs
    CMB_tracer = ccl.CMBLensingTracer(cosmology, z_source=z_CMB)

    # define common range of multipoles and a dictionary
    ell_values = np.logspace(np.log10(2), np.log10(3000), 100)
    cl_spectra = {}

    # generate a colormap for distinct colors
    colors = cm.get_cmap('tab20', num_lens_bins ** 2)
    color_idx = 0

    # Lens galaxy-Lens galaxy auto-corr spectra
    if 'GG' in correlation_types:
        for i in range(num_lens_bins):
            for k in range(i, num_lens_bins): # Avoid duplicates, i.e., LL_1_2 is same as LL_2_1
                cl = cosmology.angular_cl(lens_tracers_nc[i], lens_tracers_nc[k], ell_values, p_of_k_a = Pk2D_object)    
                cl_spectra[f'GG{i+1}_{k+1}'] = cl


    # galaxy lensing-galaxy lensing auto-corr spectra
    if 'LL' in correlation_types:
        for i in range(num_source_bins):
            for k in range(i, num_source_bins): # Avoid duplicates, i.e., LL_1_2 is same as LL_2_1
                cl = cosmology.angular_cl(lensing_tracers_nc[i], lensing_tracers_nc[k], ell_values, p_of_k_a = Pk2D_object)
                cl_spectra[f'LL{i+1}_{k+1}'] = cl


    # galaxy-galaxy lensing cross-corr spectra
    if 'GL' in correlation_types:
        for i in range (num_source_bins):
            for k in range(num_lens_bins):
                cl = cosmology.angular_cl(lens_tracers_nc[k], lensing_tracers_nc[i], ell_values, p_of_k_a = Pk2D_object)
                cl_spectra[f'GL{i+1}_{k+1}'] = cl

    # CMB lensing-galaxy lensing cross-corr spectra
    if 'CL' in correlation_types:
        for j in range(num_source_bins):
            cl = cosmology.angular_cl(lensing_tracers_nc[j], CMB_tracer, ell_values, p_of_k_a = Pk2D_object)
            cl_spectra[f'CL{j+1}'] = cl

    # CMB lensing-lens galaxies cross-corr spectra
    if 'CG' in correlation_types:
        for j in range(num_lens_bins):
            cl = cosmology.angular_cl(lens_tracers_nc[j], CMB_tracer, ell_values, p_of_k_a = Pk2D_object)
            cl_spectra[f'CG{j+1}'] = cl

    # CMB lensing-CMB lensing auto-corr spectra
    if 'CC' in correlation_types:
        cl = cosmology.angular_cl(CMB_tracer, CMB_tracer, ell_values, p_of_k_a = Pk2D_object)
        cl_spectra[f'CC'] = cl

    if plot.lower() == 'yes':
        # plot all calculated Cl spectra
        plt.figure(figsize=(12, 8))
        for key, cl_values in cl_spectra.items():
            plt.loglog(ell_values, cl_values, label=key, color=colors(color_idx % colors.N))
            color_idx += 1

        plt.xlabel(r'Multipole, $\ell$')
        plt.ylabel(r'Angular Power Spectrum, $C_\ell$')
        plt.title(r'Angular Power Spectra ($C_\ell$) for ' + ', '.join(correlation_types) + ' Correlations')
        plt.legend(loc='best', fontsize='small', bbox_to_anchor=(1.05, 1))
        plt.grid(True, which="both", ls="-")
        plt.tight_layout()
        plt.show()

    elif plot.lower() == 'no':
        return cl_spectra

    else:
        print(f"Warning: Invalid plot option '{plot}'. Expected 'yes' or 'no'. Proceeding without plotting.")
   
# compute general spectra with noise
def build_tracers_from_data(cosmo, lens_data, source_data, magnification_bias_lenses=None):
    # build CCL tracers from distributions
    z_lens = lens_data[:, 0]
    z_source = source_data[:, 0]

    n_lens = lens_data.shape[1] - 1
    n_src = source_data.shape[1] - 1

    lens_tracers = []
    for i in range(1, n_lens + 1):
        nz = lens_data[:, i]
        bias = np.ones_like(z_lens)
        # Use magnification bias for lens galaxies if provided
        mag_bias_values = None
        if magnification_bias_lenses is not None:
            mag_bias_values = magnification_bias_lenses * np.ones_like(z_lens)

        tracer = ccl.NumberCountsTracer(cosmo, has_rsd=False,
                                        dndz=(z_lens, nz),
                                        bias=(z_lens, bias),
                                        mag_bias=(z_lens, mag_bias_values) if mag_bias_values is not None else None)
        lens_tracers.append(tracer)

    lensing_tracers = []
    for i in range(1, n_src + 1):
        nz = source_data[:, i]
        # WeakLensingTracer does not have a 'mag_bias' parameter for number counts
        tracer = ccl.WeakLensingTracer(cosmo, dndz=(z_source, nz))
        lensing_tracers.append(tracer)

    cmb_tracer = ccl.CMBLensingTracer(cosmo, z_source=1090)

    return lens_tracers, lensing_tracers, cmb_tracer

# build tracer dictionary
def build_tracer_dict(lens_tracers, lensing_tracers, cmb_tracer):
    tracer_dict = {'phi': cmb_tracer}

    for i, tr in enumerate(lens_tracers):
        tracer_dict[f'g{i+1}'] = tr
    for i, tr in enumerate(lensing_tracers):
        tracer_dict[f'l{i+1}'] = tr

    return tracer_dict

# build noise dictionary
def build_noise_dict(f_map, ells, shot_noise_lens=None, shape_noise_source=None, cmb_noise_phi=None):
    noise_dict = {}

    if shot_noise_lens is not None:
        # assuming shot_noise_lens is a list/array with noise for each lens bin (scalar values)
        for i in range(1, len(shot_noise_lens) + 1):
            # Make it an ell-dependent array for consistent addition
            noise_dict[(f'g{i}', f'g{i}')] = np.ones_like(ells) * shot_noise_lens[i-1]

    if shape_noise_source is not None:
        # assuming shape_noise_source is a list/array with noise for each source bin (scalar values)
        for i in range(1, len(shape_noise_source) + 1):
            # Make it an ell-dependent array for consistent addition
            noise_dict[(f'l{i}', f'l{i}')] = np.ones_like(ells) * shape_noise_source[i-1]

    if cmb_noise_phi is not None:
        # assuming cmb_noise_phi is already an ell-dependent array
        noise_dict[('phi','phi')] = cmb_noise_phi

    return noise_dict

# build spectra dictionary
def build_spectra_dict_old(cosmo, f_map, tracer_dict, ells, noise_dict=None):
    spectra_dict = {}

    # Iterate through all unique pairs of tracers in tracer_dict to calculate Cls
    tracer_labels = list(tracer_dict.keys())
    for i, label1 in enumerate(tracer_labels):
        for j, label2 in enumerate(tracer_labels):
            # ensure consistent key ordering (e.g., ('g1', 'l1') not ('l1', 'g1'))
            # only calculate each unique pair once
            key_fwd = (label1, label2)
            key_bwd = (label2, label1)

            if key_fwd in spectra_dict or key_bwd in spectra_dict: # spectra already handled
                continue
            else:
                tracer1 = tracer_dict[label1]
                tracer2 = tracer_dict[label2]
                Cl = ccl.angular_cl(cosmo, tracer1, tracer2, ells)

            # store with canonical ordering
            if label1 < label2: # Simple lexicographical order for consistency
                spectra_dict[(label1, label2)] = Cl
            else:
                spectra_dict[(label2, label1)] = Cl # Store with smaller label first

    # add noise terms
    if noise_dict is not None:
        for key_noise, noise_val in noise_dict.items():
            # only add noise to auto-spectra.
            if key_noise[0] == key_noise[1]:
                spectra_dict[key_noise] += noise_val

    return spectra_dict

# build spectra dictionary w/ or w/o emulator
def build_spectra_dict(cosmo, f_map, tracer_dict, ells, noise_dict = None, linear_emulator = None, boost_emulator = None):
    spectra_dict = {}

    # make Pk2D object if an emulator is present
    if linear_emulator != None:
        a_grid = np.linspace(1/(1+5), 1.0, 20)
        z_grid = (1.0 / a_grid) - 1.0
        Pk2D_object = make_Pk2D(cosmo, linear_emulator = linear_emulator, boost_emulator = boost_emulator, z_arr = z_grid, cmin = 3.13, eta_0 = 0.60)

    # Iterate through all unique pairs of tracers in tracer_dict to calculate Cls
    tracer_labels = list(tracer_dict.keys())
    for i, label1 in enumerate(tracer_labels):
        for j, label2 in enumerate(tracer_labels):
            # ensure consistent key ordering (e.g., ('g1', 'l1') not ('l1', 'g1'))
            # only calculate each unique pair once
            key_fwd = (label1, label2)
            key_bwd = (label2, label1)

            if key_fwd in spectra_dict or key_bwd in spectra_dict: # spectra already handled
                continue
            else:
                tracer1 = tracer_dict[label1]
                tracer2 = tracer_dict[label2]

                if linear_emulator == None:
                    Cl = ccl.angular_cl(cosmo, tracer1, tracer2, ells)

                else:
                    Cl = ccl.angular_cl(cosmo, tracer1, tracer2, ells, p_of_k_a = Pk2D_object)

            # store with canonical ordering
            if label1 < label2: # Simple lexicographical order for consistency
                spectra_dict[(label1, label2)] = Cl
            else:
                spectra_dict[(label2, label1)] = Cl # Store with smaller label first

    # add noise terms
    if noise_dict is not None:
        for key_noise, noise_val in noise_dict.items():
            # only add noise to auto-spectra.
            if key_noise[0] == key_noise[1]:
                spectra_dict[key_noise] += noise_val

    return spectra_dict
    
# helper function to generate desired pairs based on simplified input
def create_simplified_desired_pairs(n_lens_bins, n_source_bins, desired_spectra):

    all_pairs = []

    def _canonicalize_pair(p1, p2):
        # Ensures consistent ordering, e.g., ('phi', 'g1') instead of ('g1', 'phi')
        # This matches the logic in ForecastMap._process_desired_pairs
        return (p1, p2) if p1 < p2 else (p2, p1)

    if 'CC' in desired_spectra:
        all_pairs.append(('phi', 'phi'))

    if 'GG' in desired_spectra:
        for i in range(1, n_lens_bins + 1):
            for j in range(i, n_lens_bins + 1):
                all_pairs.append(_canonicalize_pair(f'g{i}', f'g{j}'))

    if 'LL' in desired_spectra:
        for i in range(1, n_source_bins + 1):
            for j in range(i, n_source_bins + 1):
                all_pairs.append(_canonicalize_pair(f'l{i}', f'l{j}'))

    if 'GL' in desired_spectra:
        for i in range(1, n_source_bins + 1): # lens bins first, then lensing bins for cross
            for j in range(1, n_lens_bins + 1):
                all_pairs.append(_canonicalize_pair(f'g{i}', f'l{j}'))

    if 'CG' in desired_spectra: # Lens Galaxy-CMB Lensing (from Lens galaxies to CMB lensing)
        for i in range(1, n_lens_bins + 1):
            all_pairs.append(_canonicalize_pair(f'g{i}', 'phi'))

    if 'CL' in desired_spectra: # CMB-Source Lensing (from CMB lensing to Source galaxies)
        for i in range(1, n_source_bins + 1):
            all_pairs.append(_canonicalize_pair(f'l{i}', 'phi'))

    # Remove duplicates (though with the current logic, there shouldn't be any)
    # and ensure it's a list of tuples
    print(list(sorted(list(set(all_pairs)))))
    return list(sorted(list(set(all_pairs)))) # Sort for deterministic order

def build_covariance_from_data_old(
    cosmo,
    lens_data,
    source_data,
    f_sky,
    n_ell=3000, # This now represents the *maximum* ell for the unbinned calculation
    binsize=1,  # New parameter for binning
    shot_noise_lens=None,
    shape_noise_source=None,
    cmb_noise_phi=None,
    magnification_bias_lenses=None, # Renamed parameter for lens magnification bias (s value)
    desired_spectra=None
):

    # ForecastMap still works with the total number of ell values (unbinned)
    if desired_spectra:
        desired_pairs = create_simplified_desired_pairs(lens_data.shape[1]-1, source_data.shape[1]-1, desired_spectra)
    else:
        desired_pairs = None

    f_map = ForecastMap(n_lens=lens_data.shape[1]-1, n_src=source_data.shape[1]-1, n_ell=n_ell, desired_pairs=desired_pairs)

    # Use the full range of unbinned ells for CCL calculations
    ells = np.arange(2, n_ell + 2)

    # build spectra
    lens_tracers, source_tracers, cmb_tracer = build_tracers_from_data(cosmo, lens_data, source_data, magnification_bias_lenses)
    tracer_dict = build_tracer_dict(lens_tracers, source_tracers, cmb_tracer)
    noise_dict = build_noise_dict(f_map, ells, shot_noise_lens, shape_noise_source, cmb_noise_phi)
    spectra_dict = build_spectra_dict(cosmo, f_map, tracer_dict, ells, noise_dict)

    # build covariance -- now pass the binsize to CovarianceMatrix
    cov = CovarianceMatrix(f_map, spectra_dict, f_sky, binsize=binsize)

    return cov, spectra_dict, f_map
    
# build covariance matrix w or w/o emulator (full unless otherwise specified)
def build_covariance_from_data(
    cosmo,
    lens_data,
    source_data,
    f_sky,
    n_ell=3000, # This now represents the *maximum* ell for the unbinned calculation
    binsize=1,  # New parameter for binning
    shot_noise_lens=None,
    shape_noise_source=None,
    cmb_noise_phi=None,
    magnification_bias_lenses=None, # Renamed parameter for lens magnification bias (s value)
    desired_spectra=None):
#    linear_emulator=None,
#    boost_emulator=None
#):

    # ForecastMap still works with the total number of ell values (unbinned)
    if desired_spectra:
        desired_pairs = create_simplified_desired_pairs(lens_data.shape[1]-1, source_data.shape[1]-1, desired_spectra)
    else:
        desired_pairs = None

    f_map = ForecastMap(n_lens=lens_data.shape[1]-1, n_src=source_data.shape[1]-1, n_ell=n_ell, desired_pairs=desired_pairs)

    # Use the full range of unbinned ells for CCL calculations
    ells = np.arange(2, n_ell + 2)

    # build spectra
    lens_tracers, source_tracers, cmb_tracer = build_tracers_from_data(cosmo, lens_data, source_data, magnification_bias_lenses)
    tracer_dict = build_tracer_dict(lens_tracers, source_tracers, cmb_tracer)
    noise_dict = build_noise_dict(f_map, ells, shot_noise_lens, shape_noise_source, cmb_noise_phi)
    #spectra_dict = build_spectra_dict(cosmo, f_map, tracer_dict, ells, noise_dict, linear_emulator=None, boost_emulator=None)

    spectra_dict = build_spectra_dict_old(cosmo, f_map, tracer_dict, ells, noise_dict)

    # build covariance -- now pass the binsize to CovarianceMatrix
    cov = CovarianceMatrix(f_map, spectra_dict, f_sky, binsize=binsize)

    return cov, spectra_dict, f_map

# plot the covariance matrix, or a subset thereof
# if no specific desired spectra are given, the whole matrix will be plotted
#### check this
def plot_covariance_matrix(
    cov_obj,
    spectra_dict,
    f_map,
    binsize=1, # New parameter for binning
    desired_spectra = None,
    title='Subset of Gaussian Covariance Matrix'
):

    # determine which spectra to plot
    if desired_spectra is None:
        pairs_to_plot = f_map.pairs
    else:
        # Canonicalize the desired_spectra based on f_map's internal ordering
        # This ensures consistent lookup in the covariance matrix
        desired_pairs = create_simplified_desired_pairs(n_lens_bins=f_map.n_lens, n_source_bins=f_map.n_src, desired_spectra=desired_spectra)
        processed_desired_pairs = []
        for p in desired_pairs:
            if not isinstance(p, tuple) or len(p) != 2:
                raise ValueError(f"Each desired pair must be a tuple of two strings: {p}")
            # Apply canonical ordering logic similar to ForecastMap._process_desired_pairs
            if p[0] > p[1]:
                canonical_pair = (p[1], p[0])
            else:
                canonical_pair = p
            if canonical_pair not in processed_desired_pairs:
                processed_desired_pairs.append(canonical_pair)
        pairs_to_plot = processed_desired_pairs

    # Calculate the effective number of binned ell values for plotting
    n_ell_binned = int(np.ceil(f_map.n_ell / binsize))

    # determine dimensions for the subset matrix
    num_desired = len(pairs_to_plot)
    total_dim = num_desired * n_ell_binned
    subset_matrix = np.zeros((total_dim, total_dim))

    # populate the subset matrix
    for i in range(num_desired):
        pair_A = pairs_to_plot[i]
        for j in range(num_desired):
            pair_B = pairs_to_plot[j]
            # get the N_ell_binned x N_ell_binned block from the cov_obj
            block = cov_obj.get_block(pair_A, pair_B)
            # place it into the subset_matrix
            subset_matrix[i * n_ell_binned : (i + 1) * n_ell_binned,
                          j * n_ell_binned : (j + 1) * n_ell_binned] = block

    # plotting
    plt.figure(figsize=(12, 10))
    im = plt.imshow(subset_matrix, cmap='viridis', origin='lower',
                    extent=[0, total_dim, 0, total_dim], # extent for proper aspect ratio/labels
                    norm=LogNorm() # use LogNorm for better visualization of potentially wide range of values
                    )

    # create tick positions and labels for blocks
    tick_positions = []
    tick_labels = []

    for k in range(num_desired):
        tick_positions.append(k * n_ell_binned + n_ell_binned / 2)
        label_a, label_b = pairs_to_plot[k]

        # format labels nicely, handling CMB specific ones and numerical ones
        if label_a in ['T', 'E'] and label_b in ['T', 'E']:
            tick_labels.append(fr'$C^{{{label_a}{label_b}}}$')

        elif label_a == 'phi' and label_b == 'phi':
            tick_labels.append(fr'$C^{{\phi\phi}}$')

        elif label_a.startswith('g') and label_b.startswith('g'):
            sa = label_a.replace('g', 'g_')
            sb = label_b.replace('g', 'g_')
            tick_labels.append(fr'$C^{{{sa}{sb}}}$')

        elif label_a.startswith('l') and label_b.startswith('l'):
            sa = label_a.replace('l', 'l_')
            sb = label_b.replace('l', 'l_')
            tick_labels.append(fr'$C^{{{sa}{sb}}}$')

        elif (label_a.startswith('g') or label_a.a.startswith('l')) and label_b == 'phi':
            sa = label_a.replace('g', 'g_').replace('l', 'l_')
            tick_labels.append(fr'$C^{{{sa}\phi}}$')

        elif (label_b.startswith('g') or label_b.startswith('l')) and label_a == 'phi':
            sb = label_b.replace('g', 'g_').replace('l', 'l_')
            tick_labels.append(fr'$C^{{\phi{sb}}}$')

        else:  # lens-lensing, or general case
            sa = label_a.replace('g', 'g_').replace('l', 'l_')
            sb = label_b.replace('g', 'g_').replace('l', 'l_')
            tick_labels.append(fr'$C^{{{sa}{sb}}}$')

    plt.xticks(tick_positions, tick_labels, rotation=45, ha='right', fontsize=10)
    plt.yticks(tick_positions, tick_labels, fontsize=10)
    plt.xlabel('Covariance component')
    plt.ylabel('Covariance component')
    plt.title(title)
    plt.colorbar(im, label='Covariance Value (Log Scale)')
    plt.grid(False) # imshow usually doesn't need grid lines over the image itself

    # add lines to delineate the blocks
    for k in range(1, num_desired):
        plt.axvline(k * n_ell_binned, color='white', linestyle='--', linewidth=1)
        plt.axhline(k * n_ell_binned, color='white', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.show()

    return subset_matrix

# plot a specific subset of the covariance matrix as a correlation matrix
def plot_correlation_matrix(
    cov_obj,
    spectra_dict,
    f_map,
    binsize=1, # New parameter for binning
    desired_spectra = None,
    title='Subset of Gaussian Correlation Matrix'
):

    # determine which spectra to plot
    if desired_spectra is None:
        pairs_to_plot = f_map.pairs
    else:
        # Canonicalize the desired_spectra based on f_map's internal ordering
        # This ensures consistent lookup in the covariance matrix
        desired_pairs = create_simplified_desired_pairs(n_lens_bins=f_map.n_lens, n_source_bins=f_map.n_src, desired_spectra=desired_spectra)
        processed_desired_pairs = []
        for p in desired_pairs:
            if not isinstance(p, tuple) or len(p) != 2:
                raise ValueError(f"Each desired pair must be a tuple of two strings: {p}")
            # Apply canonical ordering logic similar to ForecastMap._process_desired_pairs
            if p[0] > p[1]:
                canonical_pair = (p[1], p[0])
            else:
                canonical_pair = p
            if canonical_pair not in processed_desired_pairs:
                processed_desired_pairs.append(canonical_pair)
        pairs_to_plot = processed_desired_pairs

    # Calculate the effective number of binned ell values for plotting
    n_ell_binned = int(np.ceil(f_map.n_ell / binsize))

    # determine dimensions for the subset matrix
    num_desired = len(pairs_to_plot)
    total_dim = num_desired * n_ell_binned
    subset_covariance_matrix = np.zeros((total_dim, total_dim))

    # populate the subset covariance matrix
    for i in range(num_desired):
        pair_A = pairs_to_plot[i]
        for j in range(num_desired):
            pair_B = pairs_to_plot[j]
            # get the N_ell_binned x N_ell_binned block from the cov_obj
            block = cov_obj.get_block(pair_A, pair_B)
            # place it into the subset_covariance_matrix
            subset_covariance_matrix[i * n_ell_binned : (i + 1) * n_ell_binned,
                                   j * n_ell_binned : (j + 1) * n_ell_binned] = block

    # Calculate the correlation matrix
    # Ensure that diagonal elements are non-zero before division
    diagonal = np.sqrt(np.diag(subset_covariance_matrix))
    # Replace zeros in diagonal with a small number to avoid division by zero
    diagonal[diagonal == 0] = 1e-10  # A small epsilon
    correlation_matrix = subset_covariance_matrix / np.outer(diagonal, diagonal)

    # Create a mask for values exactly equal to zero
    # mask = (correlation_matrix == 0)
    mask = None

    # plotting with Seaborn
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap='viridis', vmin=-1, vmax=1, square=True,
                cbar_kws={'label': 'Correlation Value'}, mask=mask)

    # create tick positions and labels for blocks
    tick_positions = []
    tick_labels = []

    for k in range(num_desired):
        tick_positions.append(k * n_ell_binned + n_ell_binned / 2)
        label_a, label_b = pairs_to_plot[k]
        # format labels nicely, handling CMB specific ones and numerical ones
        if label_a == 'phi' and label_b == 'phi':
            tick_labels.append(r'$C^{\phi\phi}$')

        elif label_a.startswith('g') and label_b.startswith('g'):
            sa = label_a.replace('g', 'g_')
            sb = label_b.replace('g', 'g_')
            tick_labels.append(fr'$C^{{{sa}{sb}}}$')

        elif label_a.startswith('l') and label_b.startswith('l'):
            sa = label_a.replace('l', 'l_')
            sb = label_b.replace('l', 'l_')
            tick_labels.append(fr'$C^{{{sa}{sb}}}$')

        elif (label_a.startswith('g') or label_a.startswith('l')) and label_b == 'phi':
            sa = label_a.replace('g', 'g_').replace('l', 'l_')
            tick_labels.append(fr'$C^{{{sa}\phi}}$')

        elif (label_b.startswith('g') or label_b.startswith('l')) and label_a == 'phi':
            sb = label_b.replace('g', 'g_').replace('l', 'l_')
            tick_labels.append(fr'$C^{{\phi{sb}}}$')

        else:  # lens-lensing, or general case
            sa = label_a.replace('g', 'g_').replace('l', 'l_')
            sb = label_b.replace('g', 'g_').replace('l', 'l_')
            tick_labels.append(fr'$C^{{{sa}{sb}}}$')

    plt.xticks(tick_positions, tick_labels, rotation=45, ha='right', fontsize=10)
    plt.yticks(tick_positions, tick_labels, fontsize=10)
    plt.xlabel('Correlation Component')
    plt.ylabel('Correlation Component')
    plt.title(title)

    # Add lines to delineate the blocks (on top of the heatmap)
    for k in range(1, num_desired):
        plt.axvline(k * n_ell_binned, color='white', linestyle='--', linewidth=1)
        plt.axhline(k * n_ell_binned, color='white', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.show()

    return correlation_matrix

# helper function to plot spectra from the spectra_dict
def plot_spectra_from_dict(spectra_dict, title_prefix='Angular Power Spectrum', desired_spectra=None):
    if not spectra_dict:
        print("No spectra to plot.")
        return

    # Get the ells from the first entry in spectra_dict
    # Assuming all spectra have the same ell values
    first_key = next(iter(spectra_dict))
    ells = np.arange(2, len(spectra_dict[first_key]) + 2)

    # Determine which spectra to plot
    spectra_to_plot_filtered = {}
    if desired_spectra is None:
        spectra_to_plot_filtered = spectra_dict # Plot all if none specified
    else:
        for d_pair in desired_spectra:
            # Canonicalize the desired pair to match spectra_dict keys
            p1, p2 = d_pair
            # Ensure consistent key ordering (e.g., ('g1', 'l1') instead of ('l1', 'g1'))
            # This should match how build_spectra_dict stores Cls (lexicographical order)
            if p1 < p2:
                canonical_pair = (p1, p2)
            else:
                canonical_pair = (p2, p1)

            if canonical_pair in spectra_dict:
                spectra_to_plot_filtered[canonical_pair] = spectra_dict[canonical_pair]
            else:
                print(f"Warning: Desired spectrum {d_pair} (canonical: {canonical_pair}) not found in spectra_dict. Skipping.")

    if not spectra_to_plot_filtered:
        print("No spectra found to plot after filtering.")
        return

    # generate a colormap with enough colors for the filtered spectra
    colors = cm.get_cmap('tab10', len(spectra_to_plot_filtered))
    color_idx = 0

    for key, cl_values in spectra_to_plot_filtered.items():
        plt.figure(figsize=(10, 6)) # New figure for each spectrum
        # Use a consistent ell range for plotting
        plt.loglog(ells, cl_values, label=f'C_l^{{{key[0]},{key[1]}}}', color=colors(color_idx % colors.N))
        color_idx += 1

        plt.xlabel(r'Multipole, $\ell$')
        plt.ylabel(r'Angular Power Spectrum, $C_\ell$')
        plt.title(f'{title_prefix} for $C_l^{{{key[0]},{key[1]}}}$')
        plt.legend(loc='best', fontsize='small')
        plt.grid(True, which="both", ls="-")
        plt.tight_layout()
        plt.show()

# get A_s from sigma8
def get_linear_As(cosmology):
    
    A_s_val = cosmology['A_s']
    if not np.isnan(A_s_val):
        return A_s_val
        
    # if A_s is missing, extract the target sigma8
    target_sigma8 = cosmology['sigma8']
    
    # instantiate a clean, minimal baseline model using a fiducial A_s
    # this captures the exact transfer function shape for these specific parameters
    fiducial_As = 2.0e-9
    
    base_params = ccl.Cosmology(
        Omega_c=cosmology['Omega_c'],
        Omega_b=cosmology['Omega_b'],
        h=cosmology['h'],
        n_s=cosmology['n_s'],
        A_s=fiducial_As,
        transfer_function='boltzmann_camb',
        matter_power_spectrum='linear'
    )
    
    # calculate the variance resulting from the baseline amplitude
    fiducial_sigma8 = ccl.sigma8(base_params)

    #### CONFIRM THIS DERIVATION
    # A_s is proportional to sigma8 ^ 2
    # exact analytical rescaling: As = As_fid * (sigma8_target / sigma8_fid)^2
    As = fiducial_As * (target_sigma8 / fiducial_sigma8) ** 2
    return As

def make_Pk2D(cosmology, linear_emulator, boost_emulator, z_arr, cmin, eta_0):
    if z_arr is None:
        a_grid = np.linspace(1/(1+5), 1.0, 20)
        z_arr = (1.0 / a_grid) - 1.0
        
    z_sorted_descending = np.sort(z_arr)[::-1]
    a_arr = 1.0 / (1.0 + z_sorted_descending)
    
    # 1. Base emulator k grid
    lk_arr_emu = np.log(linear_emulator.modes)
    
    # 2. Dense extended grid out to k ~ 1100 h/Mpc
    lk_ext = np.linspace(lk_arr_emu[-1] + 0.05, 7.0, 100)
    lk_arr = np.concatenate([lk_arr_emu, lk_ext])
    #print("lk_arr grid: ", lk_arr)
    
    pk_arr = np.zeros((len(a_arr), len(lk_arr)))

    for i, z in enumerate(z_sorted_descending):
        linear_Pk = predict_linear_Pk(cosmology, linear_emulator, z)
        
        if boost_emulator is None:
            non_linear_Pk = linear_Pk
        else:
            boost_Pk = predict_boost_Pk(cosmology, boost_emulator, z, cmin, eta_0)
            non_linear_Pk = linear_Pk * boost_Pk
        
        # 3. Use a safe, physically-motivated fixed slope for deep non-linear tails
        # stable dark matter power spectra fall off roughly as k^(-3) in this regime
        fixed_slope = -3.1 
        
        delta_lk = lk_ext - lk_arr_emu[-1]
        pk_ext = non_linear_Pk[-1] * np.exp(fixed_slope * delta_lk)
        
        pk_arr[i, :] = np.concatenate([non_linear_Pk, pk_ext])

    # 4. Build Pk2D with strict linear boundary extrapolation flags
    Pk2D = ccl.Pk2D(
        a_arr=a_arr,
        lk_arr=lk_arr,
        pk_arr=pk_arr,
        is_logp=False, 
        extrap_order_lok=1, # Strict linear extrapolation for low-k
        extrap_order_hik=1  # Strict linear extrapolation for high-k (safest)
    )
    return Pk2D
    
# create a P(k) given a general cosmology
# take in a cosmology, and emulator, and a z value and predict the P(k) using the emulator
# output a 1D array or P(k) values for given k
def predict_linear_Pk(cosmology, emulator, z):

    h = cosmology.cosmo.params.h
    h2 = h ** 2

    # cosmology objects have either sigma8 or As, but the emulator needs the latter
    A_s_val = cosmology.cosmo.params.A_s
        
    # each param must be passed as an array
    params = {
        'omega_b': np.array([cosmology.cosmo.params.Omega_b * h2]),
        'omega_cdm': np.array([cosmology.cosmo.params.Omega_c * h2]),
        'h': np.array([h]),
        'n_s': np.array([cosmology.cosmo.params.n_s]),
        'ln10^{10}A_s': np.array([np.log(A_s_val * 1e10)]),
        'z': np.array([z]),
    }

    Pk = emulator.predict(params)
    return Pk

def predict_boost_Pk(cosmology, emulator, z, cmin, eta_0):

    h = cosmology.cosmo.params.h
    h2 = h ** 2

    # cosmology objects have either sigma8 or As, but the emulator needs the latter
    A_s_val = cosmology.cosmo.params.A_s
        
    # each param must be passed as an array
    params = {
        'omega_b': np.array([cosmology.cosmo.params.Omega_b * h2]),
        'omega_cdm': np.array([cosmology.cosmo.params.Omega_c * h2]),
        'h': np.array([h]),
        'n_s': np.array([cosmology.cosmo.params.n_s]),
        'ln10^{10}A_s': np.array([np.log(A_s_val * 1e10)]),
        'cmin': np.array([cmin]),
        'eta_0': np.array([eta_0]),
        'z': np.array([z]),
    }
        
    Pk = emulator.predict(params)
    return Pk
# -------------------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------------------- #

## CLASSES
# we need to figure out what spectra and spectra pairs must be calculated for any given data set and number of ells
class ForecastMap:
    def __init__(self, n_lens=4, n_src=4, n_ell=20, desired_pairs=None):
        self.n_lens = n_lens
        self.n_src = n_src
        self.n_ell = n_ell

        # list of unique spectra
        if desired_pairs:
            self.pairs = self._process_desired_pairs(desired_pairs)
        else:
            self.pairs = self._generate_all_pairs()

        self.pair_to_index = {p: i for i, p in enumerate(self.pairs)}

        # length of data vector
        self.vector_length = len(self.pairs) * n_ell

    def _generate_all_pairs(self): # Renamed from _generate_pairs
        p = []

        # CMB spectra
        p += [('phi','phi')]

        # Lens galaxies auto/cross -- not zero indexing because of convention
        for i in range(1, self.n_lens + 1):
            for j in range(i, self.n_lens + 1):
                p.append((f'g{i}', f'g{j}'))

        # Galaxy lensing auto/cross
        for i in range(1, self.n_src + 1):
            for j in range(i, self.n_src + 1):
                p.append((f'l{i}', f'l{j}'))

        # Lens galaxies-galaxy lensingcross
        for i in range(1, self.n_lens + 1):
            for j in range(1, self.n_src + 1):
                p.append((f'g{i}', f'l{j}'))

        # Lens galaxies -- CMB lensing cross
        for i in range(1, self.n_lens + 1):
            p.append((f'g{i}', 'phi'))

        # Galaxy lensing -- CMB lensing cross
        for j in range(1, self.n_src + 1):
            p.append((f'l{j}', 'phi'))

        return p

    def _process_desired_pairs(self, desired_pairs_input):
        # Ensure consistent ordering and uniqueness for desired_pairs
        processed_pairs = []
        for pair in desired_pairs_input:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(f"Each desired pair must be a tuple of two strings: {pair}")

            # Canonical ordering: ensure the first element is \"lexicographically\" smaller
            # This aligns with how build_spectra_dict stores Cls (label1 < label2)
            if pair[0] > pair[1]:
                canonical_pair = (pair[1], pair[0])
            else:
                canonical_pair = pair

            if canonical_pair not in processed_pairs:
                processed_pairs.append(canonical_pair)
        return processed_pairs

    # this lets you find the indices of the start and end of the section with the given covariance
    def get_indices(self, pair_label):
        # Canonicalize the input pair_label for lookup in self.pair_to_index
        if pair_label[0] > pair_label[1]:
            lookup_pair = (pair_label[1], pair_label[0])
        else:
            lookup_pair = pair_label

        if lookup_pair in self.pair_to_index:
            idx = self.pair_to_index[lookup_pair]
        else:
            raise ValueError(f"Pair {pair_label} (or its canonical form {lookup_pair}) not found in the forecast map's defined pairs.")
        start = idx * self.n_ell
        end = (idx + 1) * self.n_ell
        return start, end

# build covariance matrix
# this is a class that is a massive covariance matrix
# with functions that has functions
class CovarianceMatrix:
    # initialize
    def __init__(self, f_map, spectra_dict, f_sky, binsize=1):
        self.f_map = f_map # ForecastMap object
        self.spectra_dict = spectra_dict #dictionary mapping (tracer1, tracer2) to C_l^(tracer1, tracer2) array of length n_ell
        self.f_sky = f_sky
        self.binsize = binsize

        # N_ell from ForecastMap is the original, unbinned number of ell values
        self.N_ell_unbinned = f_map.n_ell
        self.N_ell_binned = int(np.ceil(self.N_ell_unbinned / self.binsize))

        # The total length of the flattened data vector after binning
        self.N = len(f_map.pairs) * self.N_ell_binned

        # master covariance matrix dimensions are based on binned ell values
        self.matrix = np.zeros((self.N, self.N))
        self.block_slices = {}

        # build covariance
        self._build_master_covariance()

    # get the relevant C_l spectra
    def _compute_block(self, pair_A, pair_B):
        # ells from 2 to N_ell_unbinned + 1, so the indices i directly correspond to ell_values[i-2]
        ells_unbinned = np.arange(2, self.N_ell_unbinned + 2)

        a, b = pair_A
        c, d = pair_B

        def get_Cl(x, y):
            key_fwd = (x,y)
            key_bwd = (y,x)
            if key_fwd in self.spectra_dict:
                return self.spectra_dict[key_fwd]
            elif key_bwd in self.spectra_dict:
                return self.spectra_dict[key_bwd]
            else:
                # If the cross-spectrum is not explicitly calculated, assume it's zero.
                return np.zeros(self.N_ell_unbinned) # Return an array of zeros of unbinned length

        Cl_ac = get_Cl(a, c)
        Cl_bd = get_Cl(b, d)
        Cl_ad = get_Cl(a, d)
        Cl_bc = get_Cl(b, c)

        # Initialize a block for binned values
        binned_block = np.zeros((self.N_ell_binned, self.N_ell_binned))

        # Calculate covariance for each original ell, then bin
        for i_bin in range(self.N_ell_binned):
            # Determine the range of unbinned ell indices for the current bin
            # Note: ells_unbinned are indexed starting from 0, but correspond to ell=2,3,...
            start_idx_unbinned = i_bin * self.binsize # Index in the 0-indexed unbinned Cl arrays
            end_idx_unbinned = min((i_bin + 1) * self.binsize, self.N_ell_unbinned)

            if start_idx_unbinned >= self.N_ell_unbinned:
                break

            ell_indices_in_bin = np.arange(start_idx_unbinned, end_idx_unbinned)

            if len(ell_indices_in_bin) == 0:
                continue

            # Extract relevant unbinned ell values and Cls for the current bin
            current_ells_for_bin = ells_unbinned[ell_indices_in_bin]
            current_Cl_ac = Cl_ac[ell_indices_in_bin]
            current_Cl_bd = Cl_bd[ell_indices_in_bin]
            current_Cl_ad = Cl_ad[ell_indices_in_bin]
            current_Cl_bc = Cl_bc[ell_indices_in_bin]

            # Calculate the unbinned covariance terms for the diagonal elements within this bin
            # We average the (Cl_ac*Cl_bd + Cl_ad*Cl_bc) / (2ell+1) for all ell in the bin
            denom_factors = (2 * current_ells_for_bin + 1) * self.f_sky

            # Avoid division by zero
            denom_factors[denom_factors == 0] = np.inf

            # calculate each covariance value
            cov_terms_unbinned_diag = (current_Cl_ac * current_Cl_bd + current_Cl_ad * current_Cl_bc) / denom_factors

            # average covariance values in the bin
            binned_block[i_bin, i_bin] = np.mean(cov_terms_unbinned_diag)

        return binned_block

    def _build_master_covariance(self):
        for pair_A in self.f_map.pairs:
            # The indices from f_map are still for the unbinned ells.
            # We need to adapt this or handle it in get_indices for CovarianceMatrix.
            # For CovarianceMatrix, the indices sA, eA should be based on N_ell_binned

            # Recalculate block start/end indices based on binned N_ell for the master matrix
            idx_A = self.f_map.pair_to_index[pair_A]
            sA = idx_A * self.N_ell_binned
            eA = (idx_A + 1) * self.N_ell_binned

            for pair_B in self.f_map.pairs:
                idx_B = self.f_map.pair_to_index[pair_B]
                sB = idx_B * self.N_ell_binned
                eB = (idx_B + 1) * self.N_ell_binned

                block = self._compute_block(pair_A, pair_B)
                self.matrix[sA:eA, sB:eB] = block
                self.block_slices[(pair_A, pair_B)] = (slice(sA, eA), slice(sB, eB))

    # access methods now operate on the binned matrix
    def get_block(self, pair_A, pair_B):
        sA, sB = self.block_slices[(pair_A, pair_B)]
        return self.matrix[sA, sB]

    def get_value(self, pair_A, pair_B, ell_bin_idx):
        # return covariance value for two spectra and a specific binned ell index
        sA, sB = self.block_slices[(pair_A, pair_B)]
        return self.matrix[sA.start + ell_bin_idx, sB.start + ell_bin_idx]


# make SO DESI Likelihood class
class SO_x_DESI_Likelihood(Likelihood):

    params = {
        "Omega_m": None, # matter density
        #"sigma8": None,  # amplitude of matter fluctuations
        "A_s": None,      # amplitude of primordial fluctuations
        "h": None,       # Hubble parameter
        "Omega_b": None, # baryon density
        "n_s": None,     # primordial tilt
        "w0": None,      # dark energy equation of state parameter
        "wa": None,      # dark energy equation of state parameter evolution
        "Omega_k": None, # curvature density (for curved LCDM) - will set to 0 for flat_LCDM
    }

    # data-related settings, to be defined when configuring Cobaya
    data_specs: dict

    # initialize the likelihood
    # set up fiducial cosmology, calculate fiducial data vector and covariance matrix
    def initialize(self):
        print("Initializing SO_x_DESI_Likelihood...")

        # extract necessary data specifications from the Cobaya input YAML/dictionary
        # these parameters are passed via the 'data_specs' key in the Cobaya configuration.
        self.f_sky = self.data_specs.get('f_sky', 0.4) # default to 0.4 if not provided
        self.n_ell = self.data_specs.get('n_ell', 3000) # max unbinned ell
        self.binsize = self.data_specs.get('binsize', 50) # binning size for ell
        self.magnification_bias_lenses = self.data_specs.get('magnification_bias_lenses', 0.8)
        self.desired_spectra = self.data_specs.get('desired_spectra', ['GG', 'LL', 'GL', 'CC', 'CL', 'CG'])

        # noise parameters, also sourced from data_specs, with default 'None'
        self.shot_noise_lens = self.data_specs.get('shot_noise_lens', None)
        self.shape_noise_source = self.data_specs.get('shape_noise_source', None)
        self.cmb_noise_phi = self.data_specs.get('cmb_noise_phi', None)

        # retrieve lens and source data arrays dynamically.
        # these are expected to be available as global variables in the notebook
        # and their names are passed via data_specs
        self.lens_data = np.load(self.data_specs['lens_data_path'])
        self.source_data = np.load(self.data_specs['source_data_path'])

        print(f"  Loaded lens data from {self.data_specs['lens_data_path']}")
        print(f"  Loaded source data from: {self.data_specs['source_data_path']}")

        # New: Check for pre-computed data vector and covariance matrix paths
        self.data_vector_path = self.data_specs.get('data_vector_path')
        self.covariance_path = self.data_specs.get('covariance_path')

        if self.data_vector_path and self.covariance_path:
            print(f"  Loading observed data vector from: {self.data_vector_path}")
            self.observed_data_vector = np.load(self.data_vector_path)
            print(f"  Loading covariance matrix from: {self.covariance_path}")
            self.covariance_matrix = np.load(self.covariance_path)

            # Reconstruct f_map as it's still needed for model vector generation
            # This assumes that the binsize, n_ell, n_lens, n_src, and desired_spectra used to save
            # the data vector and covariance are consistent with the current data_specs.
            num_lens_bins = self.lens_data.shape[1] - 1
            num_source_bins = self.source_data.shape[1] - 1
            self.f_map = ForecastMap(n_lens=num_lens_bins, n_src=num_source_bins,
                                     n_ell=self.n_ell, desired_pairs=create_simplified_desired_pairs(num_lens_bins, num_source_bins, self.desired_spectra))

            # Verify compatibility (optional but good practice)
            num_binned_ells = int(np.ceil(self.n_ell / self.binsize))
            expected_data_len = len(self.f_map.pairs) * num_binned_ells
            if len(self.observed_data_vector) != expected_data_len:
                raise ValueError(f"Loaded data vector length ({len(self.observed_data_vector)}) does not match expected length ({expected_data_len}) based on f_map and binsize.")
            if self.covariance_matrix.shape != (expected_data_len, expected_data_len):
                raise ValueError(f"Loaded covariance matrix shape ({self.covariance_matrix.shape}) does not match expected shape ({(expected_data_len, expected_data_len)}) based on f_map and binsize.")

        else:
            # Existing logic to compute fiducial data and covariance if not pre-computed
            print("  No pre-computed data/covariance paths provided. Computing fiducial data and covariance...")
            fiducial_cosmo_input = self.data_specs.get('fiducial_cosmology_params', {})

            _Omega_c = fiducial_cosmo_input.get('Omega_c', flat_LCDM_cosmology.cosmo.Omega_c())
            _Omega_b = fiducial_cosmo_input.get('Omega_b', flat_LCDM_cosmology.cosmo.Omega_b())
            _h = fiducial_cosmo_input.get('h', flat_LCDM_cosmology.cosmo['h'])
            #_sigma8 = fiducial_cosmo_input.get('sigma8', flat_LCDM_cosmology.cosmo['sigma8'])
            _A_s = fiducial_cosmo_input.get('A_s', flat_LCDM_cosmology.cosmo['A_s'])
            _n_s = fiducial_cosmo_input.get('n_s', flat_LCDM_cosmology.cosmo['n_s'])
            _w0 = fiducial_cosmo_input.get('w0', flat_LCDM_cosmology.cosmo['w0'])
            _wa = fiducial_cosmo_input.get('wa', flat_LCDM_cosmology.cosmo['wa'])
            _Omega_k = fiducial_cosmo_input.get('Omega_k', flat_LCDM_cosmology.cosmo['Omega_k'])

            self.fiducial_cosmology = ccl.Cosmology(
                Omega_c=_Omega_c,
                Omega_b=_Omega_b,
                h=_h,
                #sigma8=_sigma8,
                A_s = _A_s,
                n_s=_n_s,
                w0=_w0,
                wa=_wa,
                Omega_k=_Omega_k,
                transfer_function='boltzmann_camb'
            )
            print(f"  Fiducial Cosmology parameters: Omega_c={_Omega_c}, Omega_b={_Omega_b}, h={_h}, A_s={_A_s}, n_s={_n_s}, w0={_w0}, wa={_wa}, Omega_k={_Omega_k}")

            self.fiducial_cosmology.compute_growth()
            
            # build the fiducial data vector (Cls) and covariance matrix
            self.cov_obj, self.fiducial_spectra_dict, self.f_map = \
                build_covariance_from_data(
                    self.fiducial_cosmology,
                    self.lens_data,
                    self.source_data,
                    f_sky=self.f_sky,
                    n_ell=self.n_ell,
                    binsize=self.binsize,
                    shot_noise_lens=self.shot_noise_lens,
                    shape_noise_source=self.shape_noise_source,
                    cmb_noise_phi=self.cmb_noise_phi,
                    magnification_bias_lenses=self.magnification_bias_lenses,
                    desired_spectra=self.desired_spectra, 
                    linear_emulator = None,
                    boost_emulator = None
                )
            self.covariance_matrix = self.cov_obj.matrix # Store the full matrix

            # flatten the fiducial Cls into a data vector 'D'
            self.observed_data_vector = np.array([])
            # note: ells_binned will be used for indexing the covariance matrix and observed data vector
            # however, the length for the loop should correspond to the expected number of bins.
            num_binned_ells = int(np.ceil(self.n_ell / self.binsize))

            for pair in self.f_map.pairs:
                # need to get the binned Cls
                # for the fiducial 'observed' data, we take the mean of unbinned Cls within each bin
                unbinned_cls = self.fiducial_spectra_dict[pair]
                binned_cls_for_pair = []
                for i in range(0, self.n_ell, self.binsize):
                    # ensure we don't go out of bounds for the unbinned_cls array
                    end_idx = min(i + self.binsize, len(unbinned_cls))
                    if i < end_idx:
                        binned_cls_for_pair.append(np.mean(unbinned_cls[i:end_idx]))
                    else:
                        # if a bin is empty (e.g., at the very end of ells if self.n_ell is not a multiple of binsize)
                        binned_cls_for_pair.append(0.0)

                # make sure the number of binned Cls matches the expected num_binned_ells
                while len(binned_cls_for_pair) < num_binned_ells:
                    binned_cls_for_pair.append(0.0) # pad with zeros or appropriate value

                self.observed_data_vector = np.concatenate((self.observed_data_vector, binned_cls_for_pair))

        # get the inverse covariance matrix and its log-determinant
        self.inv_covariance = np.linalg.inv(self.covariance_matrix)
        self.log_det_covariance = np.linalg.slogdet(self.covariance_matrix)[1]
        print("SO_x_DESI_Likelihood initialized successfully.")

    # get dictionary of required likelihood params
    def get_requirements(self):
        return {}

    def logp(self, **kwargs):
        # Cobaya passes parameters as keyword arguments. `ccl_data` contains the ccl.Cosmology object.
        #ccl_data = kwargs['CCL']
        #current_cosmology = ccl_data.get_cosmology()

        Omega_m = kwargs.get('Omega_m', kwargs.get('omega_m'))
        #Omega_m = kwargs['Omega_m']
        Omega_b = kwargs['Omega_b']
        h = kwargs['h']
        A_s = kwargs['A_s']
        n_s = kwargs['n_s']
        w0 = kwargs['w0']
        wa = kwargs['wa']
        Omega_k = kwargs['Omega_k']
        
        Omega_c = Omega_m - Omega_b

        if Omega_m < 0.1 or Omega_m > 0.6:
            print("omega_m outside of bounds")
    
        current_cosmology = ccl.Cosmology(
            Omega_c=Omega_c,
            Omega_b=Omega_b,
            h=h,
            A_s=A_s,
            n_s=n_s,
            w0=w0,
            wa=wa,
            Omega_k=Omega_k,
            transfer_function='boltzmann_camb'
        )

        current_cosmology.compute_growth()
        
        # calculate the theoretical model data vector M(theta) for the current cosmology
        ells = np.arange(2, self.n_ell + 2) # unbinned ells
        lens_tracers, source_tracers, cmb_tracer = build_tracers_from_data(
            current_cosmology, self.lens_data, self.source_data, self.magnification_bias_lenses)
        tracer_dict = build_tracer_dict(lens_tracers, source_tracers, cmb_tracer)
        noise_dict = build_noise_dict(self.f_map, ells, self.shot_noise_lens, self.shape_noise_source, self.cmb_noise_phi)
        current_spectra_dict = build_spectra_dict(current_cosmology, self.f_map, tracer_dict, ells, noise_dict, linear_emulator=None, boost_emulator=None)

        # flatten the current Cls into a model data vector 'M'
        model_data_vector = np.array([])
        num_binned_ells = int(np.ceil(self.n_ell / self.binsize))

        for pair in self.f_map.pairs:
            unbinned_cls = current_spectra_dict[pair]
            binned_cls_for_pair = []
            for i in range(0, self.n_ell, self.binsize):
                end_idx = min(i + self.binsize, len(unbinned_cls))
                if i < end_idx:
                    binned_cls_for_pair.append(np.mean(unbinned_cls[i:end_idx]))
                else:
                    binned_cls_for_pair.append(0.0)

            while len(binned_cls_for_pair) < num_binned_ells:
                binned_cls_for_pair.append(0.0)

            model_data_vector = np.concatenate((model_data_vector, binned_cls_for_pair))

        # calculate the difference vector (D - M(theta))
        difference_vector = self.observed_data_vector - model_data_vector

        # calculate the log-likelihood
        # ln L = -1/2 * (D - M)^T * C^-1 * (D - M) - 1/2 * ln|C|
        chi2 = difference_vector.dot(self.inv_covariance.dot(difference_vector))
        log_likelihood = -0.5 * chi2 - 0.5 * self.log_det_covariance

        return log_likelihood

# make likelihood class that uses emulator for calculated spectra 
# the emulator is not used for the initial covariance and data vectors 
# -- those don't take long to make and so I figure we should keep the accuracy up
# make SO DESI Likelihood class w/emulator
class SO_x_DESI_Likelihood_w_emulator(Likelihood):

    params = {
        "Omega_m": None, # matter density
        "A_s": None,      # amplitude of primordial fluctuations
        "h": None,       # Hubble parameter
        "Omega_b": None, # baryon density
        "n_s": None,     # primordial tilt
        "w0": None,      # dark energy equation of state parameter
        "wa": None,      # dark energy equation of state parameter evolution
        "Omega_k": None, # curvature density (for curved LCDM) - will set to 0 for flat_LCDM
    }

    # data-related settings, to be defined when configuring Cobaya
    data_specs: dict

    # initialize the likelihood
    # set up fiducial cosmology, calculate fiducial data vector and covariance matrix
    def initialize(self):
        print("Initializing SO_x_DESI_Likelihood...")

        # get emulators
        self.linear_emulator = CPJ(probe='mpk_lin')
        self.boost_emulator = CPJ(probe='mpk_boost')
        
        # extract necessary data specifications from the Cobaya input YAML/dictionary
        # these parameters are passed via the 'data_specs' key in the Cobaya configuration.
        self.f_sky = self.data_specs.get('f_sky', 0.4) # default to 0.4 if not provided
        self.n_ell = self.data_specs.get('n_ell', 3000) # max unbinned ell
        self.binsize = self.data_specs.get('binsize', 50) # binning size for ell
        self.magnification_bias_lenses = self.data_specs.get('magnification_bias_lenses', 0.8)
        self.desired_spectra = self.data_specs.get('desired_spectra', ['GG', 'LL', 'GL', 'CC', 'CL', 'CG'])

        # noise parameters, also sourced from data_specs, with default 'None'
        self.shot_noise_lens = self.data_specs.get('shot_noise_lens', None)
        self.shape_noise_source = self.data_specs.get('shape_noise_source', None)
        self.cmb_noise_phi = self.data_specs.get('cmb_noise_phi', None)

        # retrieve lens and source data arrays dynamically.
        # these are expected to be available as global variables in the notebook
        # and their names are passed via data_specs
        self.lens_data = np.load(self.data_specs['lens_data_path'])
        self.source_data = np.load(self.data_specs['source_data_path'])

        print(f"  Loaded lens data from {self.data_specs['lens_data_path']}")
        print(f"  Loaded source data from: {self.data_specs['source_data_path']}")

        # New: Check for pre-computed data vector and covariance matrix paths
        self.data_vector_path = self.data_specs.get('data_vector_path')
        self.covariance_path = self.data_specs.get('covariance_path')

        if self.data_vector_path and self.covariance_path:
            print(f"  Loading observed data vector from: {self.data_vector_path}")
            self.observed_data_vector = np.load(self.data_vector_path)
            print(f"  Loading covariance matrix from: {self.covariance_path}")
            self.covariance_matrix = np.load(self.covariance_path)

            # Reconstruct f_map as it's still needed for model vector generation
            # This assumes that the binsize, n_ell, n_lens, n_src, and desired_spectra used to save
            # the data vector and covariance are consistent with the current data_specs.
            num_lens_bins = self.lens_data.shape[1] - 1
            num_source_bins = self.source_data.shape[1] - 1
            self.f_map = ForecastMap(n_lens=num_lens_bins, n_src=num_source_bins,
                                     n_ell=self.n_ell, desired_pairs=create_simplified_desired_pairs(num_lens_bins, num_source_bins, self.desired_spectra))

            # Verify compatibility (optional but good practice)
            num_binned_ells = int(np.ceil(self.n_ell / self.binsize))
            expected_data_len = len(self.f_map.pairs) * num_binned_ells
            if len(self.observed_data_vector) != expected_data_len:
                raise ValueError(f"Loaded data vector length ({len(self.observed_data_vector)}) does not match expected length ({expected_data_len}) based on f_map and binsize.")
            if self.covariance_matrix.shape != (expected_data_len, expected_data_len):
                raise ValueError(f"Loaded covariance matrix shape ({self.covariance_matrix.shape}) does not match expected shape ({(expected_data_len, expected_data_len)}) based on f_map and binsize.")

        else:
            # Existing logic to compute fiducial data and covariance if not pre-computed
            print("  No pre-computed data/covariance paths provided. Computing fiducial data and covariance...")
            fiducial_cosmo_input = self.data_specs.get('fiducial_cosmology_params', {})

            _Omega_c = fiducial_cosmo_input.get('Omega_c', flat_LCDM_cosmology.cosmo.Omega_c())
            _Omega_b = fiducial_cosmo_input.get('Omega_b', flat_LCDM_cosmology.cosmo.Omega_b())
            _h = fiducial_cosmo_input.get('h', flat_LCDM_cosmology.cosmo['h'])
            _A_s = fiducial_cosmo_input.get('A_s', flat_LCDM_cosmology.cosmo['A_s'])
            _n_s = fiducial_cosmo_input.get('n_s', flat_LCDM_cosmology.cosmo['n_s'])
            _w0 = fiducial_cosmo_input.get('w0', flat_LCDM_cosmology.cosmo['w0'])
            _wa = fiducial_cosmo_input.get('wa', flat_LCDM_cosmology.cosmo['wa'])
            _Omega_k = fiducial_cosmo_input.get('Omega_k', flat_LCDM_cosmology.cosmo['Omega_k'])

            self.fiducial_cosmology = ccl.Cosmology(
                Omega_c=_Omega_c,
                Omega_b=_Omega_b,
                h=_h,
                A_s=_A_s,
                n_s=_n_s,
                w0=_w0,
                wa=_wa,
                Omega_k=_Omega_k,
                transfer_function='boltzmann_camb'
            )
            print(f"  Fiducial Cosmology parameters: Omega_c={_Omega_c}, Omega_b={_Omega_b}, h={_h}, A_s={_A_s}, n_s={_n_s}, w0={_w0}, wa={_wa}, Omega_k={_Omega_k}")

            self.fiducial_cosmology.compute_growth()
            
            # build the fiducial data vector (Cls) and covariance matrix
            self.cov_obj, self.fiducial_spectra_dict, self.f_map = \
                build_covariance_from_data(
                    self.fiducial_cosmology,
                    self.lens_data,
                    self.source_data,
                    f_sky=self.f_sky,
                    n_ell=self.n_ell,
                    binsize=self.binsize,
                    shot_noise_lens=self.shot_noise_lens,
                    shape_noise_source=self.shape_noise_source,
                    cmb_noise_phi=self.cmb_noise_phi,
                    magnification_bias_lenses=self.magnification_bias_lenses,
                    desired_spectra=self.desired_spectra,
                    linear_emulator=None,
                    boost_emulator=None
                )
            self.covariance_matrix = self.cov_obj.matrix 

            # flatten the fiducial Cls into a data vector 'D'
            self.observed_data_vector = np.array([])
            # note: ells_binned will be used for indexing the covariance matrix and observed data vector
            # however, the length for the loop should correspond to the expected number of bins.
            num_binned_ells = int(np.ceil(self.n_ell / self.binsize))

            for pair in self.f_map.pairs:
                # need to get the binned Cls
                # for the fiducial 'observed' data, we take the mean of unbinned Cls within each bin
                unbinned_cls = self.fiducial_spectra_dict[pair]
                binned_cls_for_pair = []
                for i in range(0, self.n_ell, self.binsize):
                    # ensure we don't go out of bounds for the unbinned_cls array
                    end_idx = min(i + self.binsize, len(unbinned_cls))
                    if i < end_idx:
                        binned_cls_for_pair.append(np.mean(unbinned_cls[i:end_idx]))
                    else:
                        # if a bin is empty (e.g., at the very end of ells if self.n_ell is not a multiple of binsize)
                        binned_cls_for_pair.append(0.0)

                # make sure the number of binned Cls matches the expected num_binned_ells
                while len(binned_cls_for_pair) < num_binned_ells:
                    binned_cls_for_pair.append(0.0) # pad with zeros or appropriate value

                self.observed_data_vector = np.concatenate((self.observed_data_vector, binned_cls_for_pair))

        # get the inverse covariance matrix and its log-determinant
        self.inv_covariance = np.linalg.inv(self.covariance_matrix)
        self.log_det_covariance = np.linalg.slogdet(self.covariance_matrix)[1]
        print("SO_x_DESI_Likelihood initialized successfully.")

    # get dictionary of required likelihood params
    def get_requirements(self):
        return {}

    def logp(self, **kwargs):
        # Cobaya passes parameters as keyword arguments. `ccl_data` contains the ccl.Cosmology object.
        #ccl_data = kwargs['CCL']
        #current_cosmology = ccl_data.get_cosmology()

        Omega_m = kwargs.get('Omega_m', kwargs.get('omega_m'))
        #Omega_m = kwargs['Omega_m']
        Omega_b = kwargs['Omega_b']
        h = kwargs['h']
        A_s = kwargs['A_s']
        n_s = kwargs['n_s']
        w0 = kwargs['w0']
        wa = kwargs['wa']
        Omega_k = kwargs['Omega_k']
        Omega_c = Omega_m - Omega_b

        if Omega_m < 0.1 or Omega_m > 0.6:
            print("omega_m outside of bounds")
    
        current_cosmology = ccl.Cosmology(
            Omega_c=Omega_c,
            Omega_b=Omega_b,
            h=h,
            A_s=A_s,
            n_s=n_s,
            w0=w0,
            wa=wa,
            Omega_k=Omega_k,
            transfer_function='boltzmann_camb'
        )
        
        current_cosmology.compute_growth()
        
        # calculate the theoretical model data vector M(theta) for the current cosmology
        ells = np.arange(2, self.n_ell + 2) # unbinned ells
        lens_tracers, source_tracers, cmb_tracer = build_tracers_from_data(
            current_cosmology, self.lens_data, self.source_data, self.magnification_bias_lenses)
        tracer_dict = build_tracer_dict(lens_tracers, source_tracers, cmb_tracer)
        noise_dict = build_noise_dict(self.f_map, ells, self.shot_noise_lens, self.shape_noise_source, self.cmb_noise_phi)
        current_spectra_dict = build_spectra_dict(current_cosmology, self.f_map, tracer_dict, ells, noise_dict, linear_emulator=self.linear_emulator, boost_emulator=self.boost_emulator)

        # flatten the current Cls into a model data vector 'M'
        model_data_vector = np.array([])
        num_binned_ells = int(np.ceil(self.n_ell / self.binsize))

        for pair in self.f_map.pairs:
            unbinned_cls = current_spectra_dict[pair]
            binned_cls_for_pair = []
            for i in range(0, self.n_ell, self.binsize):
                end_idx = min(i + self.binsize, len(unbinned_cls))
                if i < end_idx:
                    binned_cls_for_pair.append(np.mean(unbinned_cls[i:end_idx]))
                else:
                    binned_cls_for_pair.append(0.0)

            while len(binned_cls_for_pair) < num_binned_ells:
                binned_cls_for_pair.append(0.0)

            model_data_vector = np.concatenate((model_data_vector, binned_cls_for_pair))

        # calculate the difference vector (D - M(theta))
        difference_vector = self.observed_data_vector - model_data_vector

        # calculate the log-likelihood
        # ln L = -1/2 * (D - M)^T * C^-1 * (D - M) - 1/2 * ln|C|
        chi2 = difference_vector.dot(self.inv_covariance.dot(difference_vector))
        log_likelihood = -0.5 * chi2 - 0.5 * self.log_det_covariance

        return log_likelihood

# make SO DESI Likelihood class
class SO_x_DESI_Likelihood_sigma8_version(Likelihood):

    params = {
        "Omega_m": None, # matter density
        "sigma8": None,  # amplitude of matter fluctuations
        "h": None,       # Hubble parameter
        "Omega_b": None, # baryon density
        "n_s": None,     # primordial tilt
        "w0": None,      # dark energy equation of state parameter
        "wa": None,      # dark energy equation of state parameter evolution
        "Omega_k": None, # curvature density (for curved LCDM) - will set to 0 for flat_LCDM
    }

    # data-related settings, to be defined when configuring Cobaya
    data_specs: dict

    # initialize the likelihood
    # set up fiducial cosmology, calculate fiducial data vector and covariance matrix
    def initialize(self):
        print("Initializing SO_x_DESI_Likelihood...")

        # extract necessary data specifications from the Cobaya input YAML/dictionary
        # these parameters are passed via the 'data_specs' key in the Cobaya configuration.
        self.f_sky = self.data_specs.get('f_sky', 0.4) # default to 0.4 if not provided
        self.n_ell = self.data_specs.get('n_ell', 3000) # max unbinned ell
        self.binsize = self.data_specs.get('binsize', 50) # binning size for ell
        self.magnification_bias_lenses = self.data_specs.get('magnification_bias_lenses', 0.8)
        self.desired_spectra = self.data_specs.get('desired_spectra', ['GG', 'LL', 'GL', 'CC', 'CL', 'CG'])

        # noise parameters, also sourced from data_specs, with default 'None'
        self.shot_noise_lens = self.data_specs.get('shot_noise_lens', None)
        self.shape_noise_source = self.data_specs.get('shape_noise_source', None)
        self.cmb_noise_phi = self.data_specs.get('cmb_noise_phi', None)

        # retrieve lens and source data arrays dynamically.
        # these are expected to be available as global variables in the notebook
        # and their names are passed via data_specs
        self.lens_data = np.load(self.data_specs['lens_data_path'])
        self.source_data = np.load(self.data_specs['source_data_path'])

        print(f"  Loaded lens data from {self.data_specs['lens_data_path']}")
        print(f"  Loaded source data from: {self.data_specs['source_data_path']}")

        # New: Check for pre-computed data vector and covariance matrix paths
        self.data_vector_path = self.data_specs.get('data_vector_path')
        self.covariance_path = self.data_specs.get('covariance_path')

        if self.data_vector_path and self.covariance_path:
            print(f"  Loading observed data vector from: {self.data_vector_path}")
            self.observed_data_vector = np.load(self.data_vector_path)
            print(f"  Loading covariance matrix from: {self.covariance_path}")
            self.covariance_matrix = np.load(self.covariance_path)

            # Reconstruct f_map as it's still needed for model vector generation
            # This assumes that the binsize, n_ell, n_lens, n_src, and desired_spectra used to save
            # the data vector and covariance are consistent with the current data_specs.
            num_lens_bins = self.lens_data.shape[1] - 1
            num_source_bins = self.source_data.shape[1] - 1
            self.f_map = ForecastMap(n_lens=num_lens_bins, n_src=num_source_bins,
                                     n_ell=self.n_ell, desired_pairs=create_simplified_desired_pairs(num_lens_bins, num_source_bins, self.desired_spectra))

            # Verify compatibility (optional but good practice)
            num_binned_ells = int(np.ceil(self.n_ell / self.binsize))
            expected_data_len = len(self.f_map.pairs) * num_binned_ells
            if len(self.observed_data_vector) != expected_data_len:
                raise ValueError(f"Loaded data vector length ({len(self.observed_data_vector)}) does not match expected length ({expected_data_len}) based on f_map and binsize.")
            if self.covariance_matrix.shape != (expected_data_len, expected_data_len):
                raise ValueError(f"Loaded covariance matrix shape ({self.covariance_matrix.shape}) does not match expected shape ({(expected_data_len, expected_data_len)}) based on f_map and binsize.")

        else:
            # Existing logic to compute fiducial data and covariance if not pre-computed
            print("  No pre-computed data/covariance paths provided. Computing fiducial data and covariance...")
            fiducial_cosmo_input = self.data_specs.get('fiducial_cosmology_params', {})

            _Omega_c = fiducial_cosmo_input.get('Omega_c', flat_LCDM_cosmology.cosmo.Omega_c())
            _Omega_b = fiducial_cosmo_input.get('Omega_b', flat_LCDM_cosmology.cosmo.Omega_b())
            _h = fiducial_cosmo_input.get('h', flat_LCDM_cosmology.cosmo['h'])
            _sigma8 = fiducial_cosmo_input.get('sigma8', flat_LCDM_cosmology.cosmo['sigma8'])
            _n_s = fiducial_cosmo_input.get('n_s', flat_LCDM_cosmology.cosmo['n_s'])
            _w0 = fiducial_cosmo_input.get('w0', flat_LCDM_cosmology.cosmo['w0'])
            _wa = fiducial_cosmo_input.get('wa', flat_LCDM_cosmology.cosmo['wa'])
            _Omega_k = fiducial_cosmo_input.get('Omega_k', flat_LCDM_cosmology.cosmo['Omega_k'])

            self.fiducial_cosmology = ccl.Cosmology(
                Omega_c=_Omega_c,
                Omega_b=_Omega_b,
                h=_h,
                sigma8=_sigma8,
                n_s=_n_s,
                w0=_w0,
                wa=_wa,
                Omega_k=_Omega_k,
                transfer_function='boltzmann_camb'
            )
            print(f"  Fiducial Cosmology parameters: Omega_c={_Omega_c}, Omega_b={_Omega_b}, h={_h}, sigma8={_sigma8}, n_s={_n_s}, w0={_w0}, wa={_wa}, Omega_k={_Omega_k}")

            self.fiducial_cosmology.compute_growth()
            
            # build the fiducial data vector (Cls) and covariance matrix
            self.cov_obj, self.fiducial_spectra_dict, self.f_map = \
                build_covariance_from_data(
                    self.fiducial_cosmology,
                    self.lens_data,
                    self.source_data,
                    f_sky=self.f_sky,
                    n_ell=self.n_ell,
                    binsize=self.binsize,
                    shot_noise_lens=self.shot_noise_lens,
                    shape_noise_source=self.shape_noise_source,
                    cmb_noise_phi=self.cmb_noise_phi,
                    magnification_bias_lenses=self.magnification_bias_lenses,
                    desired_spectra=self.desired_spectra, 
                    linear_emulator = None,
                    boost_emulator = None
                )
            self.covariance_matrix = self.cov_obj.matrix # Store the full matrix

            # flatten the fiducial Cls into a data vector 'D'
            self.observed_data_vector = np.array([])
            # note: ells_binned will be used for indexing the covariance matrix and observed data vector
            # however, the length for the loop should correspond to the expected number of bins.
            num_binned_ells = int(np.ceil(self.n_ell / self.binsize))

            for pair in self.f_map.pairs:
                # need to get the binned Cls
                # for the fiducial 'observed' data, we take the mean of unbinned Cls within each bin
                unbinned_cls = self.fiducial_spectra_dict[pair]
                binned_cls_for_pair = []
                for i in range(0, self.n_ell, self.binsize):
                    # ensure we don't go out of bounds for the unbinned_cls array
                    end_idx = min(i + self.binsize, len(unbinned_cls))
                    if i < end_idx:
                        binned_cls_for_pair.append(np.mean(unbinned_cls[i:end_idx]))
                    else:
                        # if a bin is empty (e.g., at the very end of ells if self.n_ell is not a multiple of binsize)
                        binned_cls_for_pair.append(0.0)

                # make sure the number of binned Cls matches the expected num_binned_ells
                while len(binned_cls_for_pair) < num_binned_ells:
                    binned_cls_for_pair.append(0.0) # pad with zeros or appropriate value

                self.observed_data_vector = np.concatenate((self.observed_data_vector, binned_cls_for_pair))

        # get the inverse covariance matrix and its log-determinant
        self.inv_covariance = np.linalg.inv(self.covariance_matrix)
        self.log_det_covariance = np.linalg.slogdet(self.covariance_matrix)[1]
        print("SO_x_DESI_Likelihood initialized successfully.")

    # get dictionary of required likelihood params
    def get_requirements(self):
        return {}

    def logp(self, **kwargs):
        # Cobaya passes parameters as keyword arguments. `ccl_data` contains the ccl.Cosmology object.
        #ccl_data = kwargs['CCL']
        #current_cosmology = ccl_data.get_cosmology()

        Omega_m = kwargs.get('Omega_m', kwargs.get('omega_m'))
        #Omega_m = kwargs['Omega_m']
        Omega_b = kwargs['Omega_b']
        h = kwargs['h']
        sigma8 = kwargs['sigma8']
        n_s = kwargs['n_s']
        w0 = kwargs['w0']
        wa = kwargs['wa']
        Omega_k = kwargs['Omega_k']
        
        Omega_c = Omega_m - Omega_b

        if Omega_m < 0.1 or Omega_m > 0.6:
            print("omega_m outside of bounds")
    
        current_cosmology = ccl.Cosmology(
            Omega_c=Omega_c,
            Omega_b=Omega_b,
            h=h,
            sigma8=sigma8,
            n_s=n_s,
            w0=w0,
            wa=wa,
            Omega_k=Omega_k,
            transfer_function='boltzmann_camb'
        )

        current_cosmology.compute_growth()
        
        # calculate the theoretical model data vector M(theta) for the current cosmology
        ells = np.arange(2, self.n_ell + 2) # unbinned ells
        lens_tracers, source_tracers, cmb_tracer = build_tracers_from_data(
            current_cosmology, self.lens_data, self.source_data, self.magnification_bias_lenses)
        tracer_dict = build_tracer_dict(lens_tracers, source_tracers, cmb_tracer)
        noise_dict = build_noise_dict(self.f_map, ells, self.shot_noise_lens, self.shape_noise_source, self.cmb_noise_phi)
        current_spectra_dict = build_spectra_dict(current_cosmology, self.f_map, tracer_dict, ells, noise_dict, linear_emulator = None, boost_emulator = None)

        # flatten the current Cls into a model data vector 'M'
        model_data_vector = np.array([])
        num_binned_ells = int(np.ceil(self.n_ell / self.binsize))

        for pair in self.f_map.pairs:
            unbinned_cls = current_spectra_dict[pair]
            binned_cls_for_pair = []
            for i in range(0, self.n_ell, self.binsize):
                end_idx = min(i + self.binsize, len(unbinned_cls))
                if i < end_idx:
                    binned_cls_for_pair.append(np.mean(unbinned_cls[i:end_idx]))
                else:
                    binned_cls_for_pair.append(0.0)

            while len(binned_cls_for_pair) < num_binned_ells:
                binned_cls_for_pair.append(0.0)

            model_data_vector = np.concatenate((model_data_vector, binned_cls_for_pair))

        # calculate the difference vector (D - M(theta))
        difference_vector = self.observed_data_vector - model_data_vector

        # calculate the log-likelihood
        # ln L = -1/2 * (D - M)^T * C^-1 * (D - M) - 1/2 * ln|C|
        chi2 = difference_vector.dot(self.inv_covariance.dot(difference_vector))
        log_likelihood = -0.5 * chi2 - 0.5 * self.log_det_covariance

        return log_likelihood


# make SO DESI Likelihood class
class SO_x_DESI_Likelihood_original(Likelihood):

    params = {
        "Omega_m": None, # matter density
        "sigma8": None,  # amplitude of matter fluctuations
        "h": None,       # Hubble parameter
        "Omega_b": None, # baryon density
        "n_s": None,     # primordial tilt
        "w0": None,      # dark energy equation of state parameter
        "wa": None,      # dark energy equation of state parameter evolution
        "Omega_k": None, # curvature density (for curved LCDM) - will set to 0 for flat_LCDM
    }

    # data-related settings, to be defined when configuring Cobaya
    data_specs: dict

    # initialize the likelihood
    # set up fiducial cosmology, calculate fiducial data vector and covariance matrix
    def initialize(self):
        print("Initializing SO_x_DESI_Likelihood...")

        # extract necessary data specifications from the Cobaya input YAML/dictionary
        # these parameters are passed via the 'data_specs' key in the Cobaya configuration.
        self.f_sky = self.data_specs.get('f_sky', 0.4) # default to 0.4 if not provided
        self.n_ell = self.data_specs.get('n_ell', 3000) # max unbinned ell
        self.binsize = self.data_specs.get('binsize', 50) # binning size for ell
        self.magnification_bias_lenses = self.data_specs.get('magnification_bias_lenses', 0.8)
        self.desired_spectra = self.data_specs.get('desired_spectra', ['GG', 'LL', 'GL', 'CC', 'CL', 'CG'])

        # noise parameters, also sourced from data_specs, with default 'None'
        self.shot_noise_lens = self.data_specs.get('shot_noise_lens', None)
        self.shape_noise_source = self.data_specs.get('shape_noise_source', None)
        self.cmb_noise_phi = self.data_specs.get('cmb_noise_phi', None)

        # retrieve lens and source data arrays dynamically.
        # these are expected to be available as global variables in the notebook
        # and their names are passed via data_specs
        self.lens_data = np.load(self.data_specs['lens_data_path'])
        self.source_data = np.load(self.data_specs['source_data_path'])

        print(f"  Loaded lens data from {self.data_specs['lens_data_path']}")
        print(f"  Loaded source data from: {self.data_specs['source_data_path']}")

        # New: Check for pre-computed data vector and covariance matrix paths
        self.data_vector_path = self.data_specs.get('data_vector_path')
        self.covariance_path = self.data_specs.get('covariance_path')

        if self.data_vector_path and self.covariance_path:
            print(f"  Loading observed data vector from: {self.data_vector_path}")
            self.observed_data_vector = np.load(self.data_vector_path)
            print(f"  Loading covariance matrix from: {self.covariance_path}")
            self.covariance_matrix = np.load(self.covariance_path)

            # Reconstruct f_map as it's still needed for model vector generation
            # This assumes that the binsize, n_ell, n_lens, n_src, and desired_spectra used to save
            # the data vector and covariance are consistent with the current data_specs.
            num_lens_bins = self.lens_data.shape[1] - 1
            num_source_bins = self.source_data.shape[1] - 1
            self.f_map = ForecastMap(n_lens=num_lens_bins, n_src=num_source_bins,
                                     n_ell=self.n_ell, desired_pairs=create_simplified_desired_pairs(num_lens_bins, num_source_bins, self.desired_spectra))

            # Verify compatibility (optional but good practice)
            num_binned_ells = int(np.ceil(self.n_ell / self.binsize))
            expected_data_len = len(self.f_map.pairs) * num_binned_ells
            if len(self.observed_data_vector) != expected_data_len:
                raise ValueError(f"Loaded data vector length ({len(self.observed_data_vector)}) does not match expected length ({expected_data_len}) based on f_map and binsize.")
            if self.covariance_matrix.shape != (expected_data_len, expected_data_len):
                raise ValueError(f"Loaded covariance matrix shape ({self.covariance_matrix.shape}) does not match expected shape ({(expected_data_len, expected_data_len)}) based on f_map and binsize.")

        else:
            # Existing logic to compute fiducial data and covariance if not pre-computed
            print("  No pre-computed data/covariance paths provided. Computing fiducial data and covariance...")
            fiducial_cosmo_input = self.data_specs.get('fiducial_cosmology_params', {})

            _Omega_c = fiducial_cosmo_input.get('Omega_c', flat_LCDM_cosmology.cosmo.Omega_c())
            _Omega_b = fiducial_cosmo_input.get('Omega_b', flat_LCDM_cosmology.cosmo.Omega_b())
            _h = fiducial_cosmo_input.get('h', flat_LCDM_cosmology.cosmo['h'])
            _sigma8 = fiducial_cosmo_input.get('sigma8', flat_LCDM_cosmology.cosmo['sigma8'])
            _n_s = fiducial_cosmo_input.get('n_s', flat_LCDM_cosmology.cosmo['n_s'])
            _w0 = fiducial_cosmo_input.get('w0', flat_LCDM_cosmology.cosmo['w0'])
            _wa = fiducial_cosmo_input.get('wa', flat_LCDM_cosmology.cosmo['wa'])
            _Omega_k = fiducial_cosmo_input.get('Omega_k', flat_LCDM_cosmology.cosmo['Omega_k'])

            self.fiducial_cosmology = ccl.Cosmology(
                Omega_c=_Omega_c,
                Omega_b=_Omega_b,
                h=_h,
                sigma8=_sigma8,
                n_s=_n_s,
                w0=_w0,
                wa=_wa,
                Omega_k=_Omega_k,
                transfer_function='bbks'
            )
            print(f"  Fiducial Cosmology parameters: Omega_c={_Omega_c}, Omega_b={_Omega_b}, h={_h}, sigma8={_sigma8}, n_s={_n_s}, w0={_w0}, wa={_wa}, Omega_k={_Omega_k}")

            # build the fiducial data vector (Cls) and covariance matrix
            self.cov_obj, self.fiducial_spectra_dict, self.f_map = \
                build_covariance_from_data(
                    self.fiducial_cosmology,
                    self.lens_data,
                    self.source_data,
                    f_sky=self.f_sky,
                    n_ell=self.n_ell,
                    binsize=self.binsize,
                    shot_noise_lens=self.shot_noise_lens,
                    shape_noise_source=self.shape_noise_source,
                    cmb_noise_phi=self.cmb_noise_phi,
                    magnification_bias_lenses=self.magnification_bias_lenses,
                    desired_spectra=self.desired_spectra,
                    linear_emulator=None,
                    boost_emulator=None
                )
            self.covariance_matrix = self.cov_obj.matrix # Store the full matrix

            # flatten the fiducial Cls into a data vector 'D'
            self.observed_data_vector = np.array([])
            # note: ells_binned will be used for indexing the covariance matrix and observed data vector
            # however, the length for the loop should correspond to the expected number of bins.
            num_binned_ells = int(np.ceil(self.n_ell / self.binsize))

            for pair in self.f_map.pairs:
                # need to get the binned Cls
                # for the fiducial 'observed' data, we take the mean of unbinned Cls within each bin
                unbinned_cls = self.fiducial_spectra_dict[pair]
                binned_cls_for_pair = []
                for i in range(0, self.n_ell, self.binsize):
                    # ensure we don't go out of bounds for the unbinned_cls array
                    end_idx = min(i + self.binsize, len(unbinned_cls))
                    if i < end_idx:
                        binned_cls_for_pair.append(np.mean(unbinned_cls[i:end_idx]))
                    else:
                        # if a bin is empty (e.g., at the very end of ells if self.n_ell is not a multiple of binsize)
                        binned_cls_for_pair.append(0.0)

                # make sure the number of binned Cls matches the expected num_binned_ells
                while len(binned_cls_for_pair) < num_binned_ells:
                    binned_cls_for_pair.append(0.0) # pad with zeros or appropriate value

                self.observed_data_vector = np.concatenate((self.observed_data_vector, binned_cls_for_pair))

        # get the inverse covariance matrix and its log-determinant
        self.inv_covariance = np.linalg.inv(self.covariance_matrix)
        self.log_det_covariance = np.linalg.slogdet(self.covariance_matrix)[1]
        print("SO_x_DESI_Likelihood initialized successfully.")

    # get dictionary of required likelihood params
    def get_requirements(self):
        return {}

    def logp(self, **kwargs):
        # Cobaya passes parameters as keyword arguments. `ccl_data` contains the ccl.Cosmology object.
        #ccl_data = kwargs['CCL']
        #current_cosmology = ccl_data.get_cosmology()

        Omega_m = kwargs.get('Omega_m', kwargs.get('omega_m'))
        #Omega_m = kwargs['Omega_m']
        Omega_b = kwargs['Omega_b']
        h = kwargs['h']
        sigma8 = kwargs['sigma8']
        n_s = kwargs['n_s']
        w0 = kwargs['w0']
        wa = kwargs['wa']
        Omega_k = kwargs['Omega_k']
        
        Omega_c = Omega_m - Omega_b

        if Omega_m < 0.1 or Omega_m > 0.6:
            print("Omega_m out of bounds")
            
        current_cosmology = ccl.Cosmology(
            Omega_c=Omega_c,
            Omega_b=Omega_b,
            h=h,
            sigma8=sigma8,
            n_s=n_s,
            w0=w0,
            wa=wa,
            Omega_k=Omega_k,
            transfer_function='bbks'
        )
        
        # calculate the theoretical model data vector M(theta) for the current cosmology
        ells = np.arange(2, self.n_ell + 2) # unbinned ells
        lens_tracers, source_tracers, cmb_tracer = build_tracers_from_data(
            current_cosmology, self.lens_data, self.source_data, self.magnification_bias_lenses)
        tracer_dict = build_tracer_dict(lens_tracers, source_tracers, cmb_tracer)
        noise_dict = build_noise_dict(self.f_map, ells, self.shot_noise_lens, self.shape_noise_source, self.cmb_noise_phi)
        current_spectra_dict = build_spectra_dict_old(current_cosmology, self.f_map, tracer_dict, ells, noise_dict)

        # flatten the current Cls into a model data vector 'M'
        model_data_vector = np.array([])
        num_binned_ells = int(np.ceil(self.n_ell / self.binsize))

        for pair in self.f_map.pairs:
            unbinned_cls = current_spectra_dict[pair]
            binned_cls_for_pair = []
            for i in range(0, self.n_ell, self.binsize):
                end_idx = min(i + self.binsize, len(unbinned_cls))
                if i < end_idx:
                    binned_cls_for_pair.append(np.mean(unbinned_cls[i:end_idx]))
                else:
                    binned_cls_for_pair.append(0.0)

            while len(binned_cls_for_pair) < num_binned_ells:
                binned_cls_for_pair.append(0.0)

            model_data_vector = np.concatenate((model_data_vector, binned_cls_for_pair))

        # calculate the difference vector (D - M(theta))
        difference_vector = self.observed_data_vector - model_data_vector

        # calculate the log-likelihood
        # ln L = -1/2 * (D - M)^T * C^-1 * (D - M) - 1/2 * ln|C|
        chi2 = difference_vector.dot(self.inv_covariance.dot(difference_vector))
        log_likelihood = -0.5 * chi2 - 0.5 * self.log_det_covariance

        return log_likelihood

        
# class to get us information on the time taken to call the Pk2D object within PyCCL
class Pk2DTimer:
    def __init__(self):
        self.total_time_spent = 0.0
        self.call_count = 0

    def reset_timers(self):
        self.total_time_spent = 0.0
        self.call_count = 0
        print("Pk2D timers and counters reset.")

    def get_timing_info(self):
        return {"total_time_ns": self.total_time_spent, "call_count": self.call_count}

# wrapper to get us information on the time taken to call the Pk2D object within PyCCL
def instrument_Pk2D(pk2d_object):
    if not isinstance(pk2d_object, ccl.Pk2D):
        raise TypeError("pk2d_object must be an instance of ccl.Pk2D")

    # check if already instrumented (to prevent infinite recursion if called multiple times on the same object)
    if hasattr(pk2d_object, '_timer') and hasattr(pk2d_object, '_original_call_func'):
        print("Warning: Pk2D object already instrumented. Resetting timer.")
        pk2d_object.reset_timers()
        return pk2d_object

    print(f"Original Pk2D __call__ type: {type(pk2d_object.__call__)}")
    # attach a Pk2DTimer instance to the Pk2D object
    pk2d_object._timer = Pk2DTimer()

    # store the original __call__ method (bound method)
    original_call_bound = pk2d_object.__call__
    # store the underlying function of the original method
    original_call_func = original_call_bound.__func__
    # store it on the object so the new timed_call can reliably access it
    pk2d_object._original_call_func = original_call_func

    original_cosmo_attribute = getattr(original_call_bound, '_cosmo', None)

    # define a new __call__ method that includes timing
    # this new method will be bound to the pk2d_object instance later
    def timed_call(self, k, a, cosmo=None, derivative=None): # Match pyccl's signature
        print("DEBUG: Inside timed_call") 
        self._timer.call_count += 1
        start_time = time.perf_counter_ns()
        # Call the original underlying function, manually passing 'self' (which is the pk2d_object)
        # Using the stored _original_call_func to avoid closure issues.
        pk_value = self._original_call_func(self, k, a, cosmo=cosmo, derivative=derivative)
        end_time = time.perf_counter_ns()
        self._timer.total_time_spent += (end_time - start_time)
        return pk_value

    # ALWAYS set the _cosmo attribute on the new `timed_call` function object.
    # this ensures that `self.__call__._cosmo` (which becomes `timed_call._cosmo`)
    # always exists, preventing the AttributeError.
    timed_call._cosmo = original_cosmo_attribute

    # replace the __call__ method of the *instance*
    # use types.MethodType to correctly bind the new method to the instance
    pk2d_object.__call__ = types.MethodType(timed_call, pk2d_object)
    print(f"New Pk2D __call__ type after instrumentation: {type(pk2d_object.__call__)}")

    # add convenience methods directly to the pk2d_object for easier access
    pk2d_object.reset_timers = pk2d_object._timer.reset_timers
    pk2d_object.get_timing_info = pk2d_object._timer.get_timing_info

    print("Pk2D object instrumented for timing.")
    return pk2d_object