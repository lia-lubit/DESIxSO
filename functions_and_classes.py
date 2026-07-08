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
import jax
import jax.numpy as jnp
import scipy
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import linregress
from scipy.integrate import simpson
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline, interp1d

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
from matplotlib.patches import Ellipse
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display, Markdown

# inference / sampling
import cobaya
from cobaya.run import run as cobaya_run
from cobaya.likelihood import Likelihood
from cobaya_utilities import fisher
import getdist
from getdist import plots, MCSamples
from getdist.mcsamples import loadMCSamples
from getdist.gaussian_mixtures import GaussianND

print("LOADING FILE:", os.path.abspath(__file__))

# -------------------------------------------------------------------------------------------------------------------------------------------- #

### FUNCTIONS

## Plotting etc.    
def _find_key_recursive(data, target_key):
    """Recursively searches for a key in a nested dictionary/list structure."""
    if isinstance(data, dict):
        if target_key in data:
            return data[target_key]
        for key, value in data.items():
            result = _find_key_recursive(value, target_key)
            if result is not None:
                return result
    elif isinstance(data, list):
        for item in data:
            result = _find_key_recursive(item, target_key)
            if result is not None:
                return result
    return None

# plot traces and contour plots from Cobaya, including reference points
# mark if the run was incomplete, but assume it was complete unless otherwise stated
def plot_cobaya_mcmc_results(chain_dir, yaml_path, sampled_params, num_chains=4, burn_in_fraction=0.2, output_dir='plots', complete=True):
    
    # 1. Parse Fiducial Cosmology dynamically from the YAML structure
    print(f"Reading fiducial values from: {yaml_path}")
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    fiducial_block = _find_key_recursive(config, 'fiducial_cosmology_params')
    if fiducial_block is None:
        raise KeyError("Could not find 'fiducial_cosmology_params' anywhere inside the provided YAML file.")

    fiducial_vals = {}
    for p in sampled_params:
        if p in fiducial_block:
            fiducial_vals[p] = float(fiducial_block[p])
        elif p == 'Omega_m' and 'Omega_c' in fiducial_block and 'Omega_b' in fiducial_block:
            fiducial_vals['Omega_m'] = float(fiducial_block['Omega_c']) + float(fiducial_block['Omega_b'])
            print(f"Derived fiducial Omega_m = Omega_c + Omega_b = {fiducial_vals['Omega_m']:.4f}")
        elif p == 'Omega_lambda':
                # Dynamically fetch or derive Omega_m first
                if 'Omega_m' in fiducial_block:
                    om = float(fiducial_block['Omega_m'])
                elif 'Omega_c' in fiducial_block and 'Omega_b' in fiducial_block:
                    om = float(fiducial_block['Omega_c']) + float(fiducial_block['Omega_b'])
                # Fetch Omega_k if it exists, otherwise default to flat universe (0.0)
                ok = float(fiducial_block.get('Omega_k'))
                fiducial_vals['Omega_lambda'] = 1.0 - om - ok
        else:
            print(f"Warning: Could not resolve fiducial value for '{p}'. Setting default to 0.0.")
            fiducial_vals[p] = 0.0

    # 2. Process Chains, Extract Initial Points, and Apply Burn-In
    all_weights = []
    all_loglikes = []
    param_tracks = {p: [] for p in sampled_params}
    initial_points = []
    raw_chain_data = []

    print(f"Processing {num_chains} chains from: {chain_dir}")
    for i in range(num_chains):
        chain_path = os.path.join(chain_dir, f'chain_task_{i}.txt')
        if os.path.exists(chain_path):
            data = np.loadtxt(chain_path)
            raw_chain_data.append(data)
            
            pt_start = {}
            for idx, param in enumerate(sampled_params):
                col_idx = idx + 2  
                pt_start[param] = data[0, col_idx]
            initial_points.append(pt_start)
            
            burn = int(burn_in_fraction * len(data))
            all_weights.append(data[burn:, 0])
            all_loglikes.append(data[burn:, 1])
            
            for idx, param in enumerate(sampled_params):
                col_idx = idx + 2
                param_tracks[param].append(data[burn:, col_idx])
        else:
            print(f"Warning: {chain_path} not found. Skipping.")

    combined_samples = np.column_stack([np.concatenate(param_tracks[p]) for p in sampled_params])
    
    latex_labels = {
        'Omega_m': r'\Omega_\mathrm{m}',
        'Omega_b': r'\Omega_\mathrm{b}',
        'Omega_lambda': r'\Omega_\mathrm{lambda}',
        'Omega_k': r'\Omega_\mathrm{k}',
        'wa': r'w_a',
        'w0': r'w_0',
        'h': r'h',
        'A_s': r'A_\mathrm{s}',
        'n_s': r'n_\mathrm{s}',
        'Neff': r'N_\mathrm{eff}',
        'm_nu': r'm_\mathrm{nu}',
        'T_CMB': r'T_\mathrm{CMB}'
    }
    labels = [latex_labels.get(p, p) for p in sampled_params]

    samples = MCSamples(
        samples=combined_samples,
        weights=np.concatenate(all_weights),
        loglikes=np.concatenate(all_loglikes),
        names=sampled_params,
        labels=labels,
        settings={'ignore_rows': 0.0}
    )

    peaks = {}
    for param in sampled_params:
        density1D = samples.get1DDensity(param)
        peaks[param] = density1D.x[np.argmax(density1D.P)]

    # --- NAME STRIPPING & COMPLETION MODIFICATIONS ---
    os.makedirs(output_dir, exist_ok=True)
    raw_run_name = os.path.basename(os.path.normpath(chain_dir))
    
    # Clean the filename by stripping out '_likelihood' or 'likelihood'
    clean_run_name = raw_run_name.replace('_likelihood', '').replace('likelihood', '')
    display_title = clean_run_name.replace('_', ' ')

    # Append tags dynamically based on the 'complete' flag state
    if not complete:
        file_suffix = "_(incomplete)"
        title_suffix = " (incomplete)"
    else:
        file_suffix = ""
        title_suffix = ""

    # --- 3. TRIANGLE CONTOUR PLOT ---

    for param_name, latex_string in latex_labels.items():
        if samples.paramNames.hasParam(param_name):
            samples.paramNames.parWithName(param_name).label = latex_string
        
    g1 = plots.get_subplot_plotter(width_inch=2.5 * len(sampled_params))
    g1.triangle_plot(
        [samples], 
        params=sampled_params, 
        filled=True, 
        contour_colors=['darkblue'],
        title_limit=1, # this allows the printing to proceed properly
        markers=fiducial_vals
    )

    for row in range(len(sampled_params)):
        for col in range(row + 1):
            ax = g1.subplots[row, col]
            if ax is None:
                continue
            p_row = sampled_params[row]
            p_col = sampled_params[col]
            if row == col:
                ax.axvline(x=peaks[p_row], color='crimson', linestyle=':', alpha=0.8, label='MCMC Peak')
                for idx, pt in enumerate(initial_points):
                    lbl = 'Initial Points' if (row == 0 and idx == 0) else ""
                    ax.axvline(x=pt[p_row], color='darkorange', linestyle='-', alpha=0.4, linewidth=1, label=lbl)
                if row == 0:  
                    ax.legend(loc='upper right', fontsize=8)
            else:
                for idx, pt in enumerate(initial_points):
                    lbl = 'Initial Points' if (row == len(sampled_params)-1 and col == len(sampled_params)-2 and idx == 0) else ""
                    ax.scatter(pt[p_col], pt[p_row], color='darkorange', marker='x', s=40, zorder=5, alpha=0.8, label=lbl)
                if row == len(sampled_params)-1 and col == len(sampled_params)-2:
                    ax.legend(loc='upper right', fontsize=8)

    plt.suptitle(f"Marginalized Constraints: {display_title}{title_suffix}", y=1.02, fontsize=10)
    triangle_save_path = os.path.join(output_dir, f"{clean_run_name}_triangle_plot{file_suffix}.pdf")
    g1.export(triangle_save_path)
    print(f"Saved triangle plot to: {triangle_save_path}")
    plt.show()

    # --- 4. TRACE PLOTS ---
    fig, axes = plt.subplots(len(sampled_params), 1, figsize=(12, 3 * len(sampled_params)), sharex=True)
    if len(sampled_params) == 1:
        axes = [axes]

    for idx, param in enumerate(sampled_params):
        col_idx = idx + 2
        ax = axes[idx]
        
        for i, chain_data in enumerate(raw_chain_data):
            ax.plot(chain_data[:, col_idx], alpha=0.6, linewidth=0.8, label=f'Chain {i}' if idx == 0 else "")
        
        ax.axhline(y=peaks[param], color='crimson', linestyle=':', alpha=0.8, label=f'Peak: {peaks[param]:.4f}')
        ax.axhline(y=fiducial_vals[param], color='black', linestyle='--', alpha=0.6, label=f'Fiducial: {fiducial_vals[param]:.4f}')
        
        param_label = latex_labels.get(param, param)
        ax.set_ylabel(f"${param_label}$")
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
    axes[0].set_title(f"MCMC Trace Plots: {display_title}{title_suffix}", fontsize=12)
    axes[-1].set_xlabel('Step Number (Including Burn-in)')
    
    plt.tight_layout()
    trace_save_path = os.path.join(output_dir, f"{clean_run_name}_traces{file_suffix}.png")
    plt.savefig(trace_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved trace plot to: {trace_save_path}")
    plt.show()

# calcualte and (maybe) plot angular power spectra
def calculate_and_plot_Cls(
    cosmology,        # ccl.Cosmology object
    source_data,         # 2D array: z, n_z_bin1, n_z_bin2, ...
    lens_data,          # 2D array: z, n_z_bin1, n_z_bin2, ...
    correlation_types=['GG', 'LL', 'GL', 'CC', 'CL', 'CG', 'TT', 'EE', 'BB', 'TE'], # List of correlation types to plot
    Pk2D_object = None,    # Pk2D object from emulator,
    plot = True,       # chose whether to plot
    plot_scaled = False, # decide whether to plot Cl or l(l+1)C_l/2pi
    plot_linear = True,
    l_min = 2,
    n_ell = 3000
):

    ## LENSING SPECTRA
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
    galaxy_lensing_tracers_nc = []
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
            galaxy_lensing_tracers_nc.append(tracer)

    # create CMB lensing tracerrs
    cmb_lensing_tracer = ccl.CMBLensingTracer(cosmology, z_source=z_CMB)

    # define common range of multipoles and a dictionary
    ell_values = np.logspace(np.log10(2), np.log10(n_ell + 2), int(n_ell / 10))
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
                cl = cosmology.angular_cl(galaxy_lensing_tracers_nc[i], galaxy_lensing_tracers_nc[k], ell_values, p_of_k_a = Pk2D_object)
                cl_spectra[f'LL{i+1}_{k+1}'] = cl

    # galaxy-galaxy lensing cross-corr spectra
    if 'GL' in correlation_types:
        for i in range (num_source_bins):
            for k in range(num_lens_bins):
                cl = cosmology.angular_cl(lens_tracers_nc[k], galaxy_lensing_tracers_nc[i], ell_values, p_of_k_a = Pk2D_object)
                cl_spectra[f'GL{i+1}_{k+1}'] = cl

    # CMB lensing-galaxy lensing cross-corr spectra
    if 'CL' in correlation_types:
        for j in range(num_source_bins):
            cl = cosmology.angular_cl(galaxy_lensing_tracers_nc[j], cmb_lensing_tracer, ell_values, p_of_k_a = Pk2D_object)
            cl_spectra[f'CL{j+1}'] = cl

    # CMB lensing-lens galaxies cross-corr spectra
    if 'CG' in correlation_types:
        for j in range(num_lens_bins):
            cl = cosmology.angular_cl(lens_tracers_nc[j], cmb_lensing_tracer, ell_values, p_of_k_a = Pk2D_object)
            cl_spectra[f'CG{j+1}'] = cl

    # CMB lensing-CMB lensing auto-corr spectra
    if 'CC' in correlation_types:
        cl = cosmology.angular_cl(cmb_lensing_tracer, cmb_lensing_tracer, ell_values, p_of_k_a = Pk2D_object)
        cl_spectra[f'CC'] = cl

    ## PRIMARY SPECTRA
    cmb_primary_requested = [spec for spec in correlation_types if spec.upper() in ['TT', 'EE', 'BB', 'TE']]
    
    if cmb_primary_requested:
        # Extract standard physical parameters on the fly from the pyccl cosmology object
        h = cosmology['h']
        ombh2 = cosmology['Omega_b'] * (h**2)
        omch2 = cosmology['Omega_c'] * (h**2)
        A_s = cosmology['A_s']
        n_s = cosmology['n_s']
        Omega_k = cosmology['Omega_k']
        w0 = cosmology['w0']
        wa = cosmology['wa']
        Neff = cosmology['Neff']
        m_nu = cosmology['m_nu']
        T_CMB = cosmology['T_CMB']
        l_max = n_ell + 2

        # fix the error wherein it thinks m_nu etc is a list
        m_nu = float(m_nu[0]) if isinstance(m_nu, (list, np.ndarray)) else float(m_nu)
        Neff = float(Neff[0]) if isinstance(Neff, (list, np.ndarray)) else float(Neff)
        T_CMB = float(T_CMB[0]) if isinstance(T_CMB, (list, np.ndarray)) else float(T_CMB)

        pars = camb.CAMBparams()
        pars.set_cosmology(H0=h*100, ombh2=ombh2, omch2=omch2, omk=Omega_k, mnu=m_nu, nnu=Neff, TCMB=T_CMB)
        pars.InitPower.set_params(As=A_s, ns=n_s)
        pars.set_dark_energy(w=w0, wa=wa, dark_energy_model='ppf')
        pars.set_for_lmax(l_max, lens_potential_accuracy=0)

        # return raw Cl not scaled Cl -- my code generally expects the raw values
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
        lensed_cls = powers['total']  
        cmb_map = {'TT': 0, 'EE': 1, 'BB': 2, 'TE': 3}
        
        for spec in cmb_primary_requested:
            idx = cmb_map[spec.upper()]
            cl_spectra[spec.upper()] = lensed_cls[ell_values.astype(int), idx]
    
    # plot
    if plot:
        plt.figure(figsize=(12, 8))
        color_idx = 0
        
        valid = (ell_values >= l_min)
        ell_filtered = ell_values[valid]
        
        for key, cl_values in cl_spectra.items():
            if plot_scaled:
                scaled_factor = ell_values * (ell_values + 1) / (2 * np.pi)
                y_values = cl_values * scaled_factor
                ylabel = r'$D_\ell = \ell(\ell+1)C_\ell / 2\pi$'
            else:
                y_values = cl_values
                ylabel = r'Angular Power Spectrum, $C_\ell$'

            plt.plot(ell_filtered, y_values[valid], label=key, color=colors(color_idx % colors.N))
            color_idx += 1

        if plot_linear:
            plt.xscale('linear')
            plt.yscale('linear')
        else:
            plt.xscale('log')
            plt.yscale('log')

        plt.xlim(2, n_ell)
        plt.xlabel(r'Multipole, $\ell$')
        plt.ylabel(ylabel)
        
        if plot_scaled:
            plt.title(r'Scaled Angular Power Spectra ($D_\ell$) for ' + ', '.join(correlation_types) + ' Correlations')
        else:
            plt.title(r'Angular Power Spectra ($C_\ell$) for ' + ', '.join(correlation_types) + ' Correlations')
            
        plt.legend(loc='best', fontsize='small', bbox_to_anchor=(1.05, 1))
        plt.grid(True, which="both", alpha=0.3)
        plt.show()
        return cl_spectra
    else:
        return cl_spectra
        
# plot the covariance matrix, or a subset thereof
# if no specific desired spectra are given, the whole matrix will be plotted
#### CHECK
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
        for p in f_map.pairs: 
            if p in desired_pairs:
                if not isinstance(p, tuple) or len(p) != 2:
                    raise ValueError(f"Each desired pair must be a tuple of two strings: {p}")
                # Apply canonical ordering logic similar to ForecastMap._process_desired_pairs
                if p[0] > p[1]:
                    canonical_pair = (p[1], p[0])
                else:
                    canonical_pair = p
                if canonical_pair not in processed_desired_pairs:
                    processed_desired_pairs.append(canonical_pair)
        pairs_to_plot = [pair for pair in f_map.pairs if pair in processed_desired_pairs]
        pairs_to_plot = list(dict.fromkeys(pairs_to_plot))

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
#            block = cov_obj.get_block(pair_B, pair_A)
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

        elif label_a == 'kappa_c' and label_b == 'kappa_c':
            tick_labels.append(fr'$C^{{\kappa_c\kappa_c}}$')

        elif label_a.startswith('g') and label_b.startswith('g'):
            sa = label_a.replace('g', 'g_')
            sb = label_b.replace('g', 'g_')
            tick_labels.append(fr'$C^{{{sa}{sb}}}$')

        elif label_a.startswith('kappa_g') and label_b.startswith('kappa_g'):
            sa = label_a.replace('kappa_g', 'kappa_g_')
            sb = label_b.replace('kappa_g', 'kappa_g_')
            tick_labels.append(fr'$C^{{{sa}{sb}}}$')

        elif (label_a.startswith('g') or label_a.a.startswith('kappa_g')) and label_b == 'kappa_c':
            sa = label_a.replace('g', 'g_').replace('kappa_g', 'kappa_g_')
            tick_labels.append(fr'$C^{{{sa}\kappa_c}}$')

        elif (label_b.startswith('g') or label_b.startswith('kappa_g')) and label_a == 'kappa_c':
            sb = label_b.replace('g', 'g_').replace('kappa_g', 'kappa_g_')
            tick_labels.append(fr'$C^{{\kappa_c{sb}}}$')

        else:  # lens-lensing, or general case
            sa = label_a.replace('g', 'g_').replace('kappa_g', 'kappa_g_')
            sb = label_b.replace('g', 'g_').replace('kappa_g', 'kappa_g_')
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
    diagonal[diagonal == 0] = 1e-30  # A small epsilon
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
        if label_a == 'kappa_c' and label_b == 'kappa_c':
            tick_labels.append(r'$C^{\kappa_c\kappa_c}$')

        elif label_a.startswith('g') and label_b.startswith('g'):
            sa = label_a.replace('g', 'g_')
            sb = label_b.replace('g', 'g_')
            tick_labels.append(fr'$C^{{{sa}{sb}}}$')

        elif label_a.startswith('kappa_g') and label_b.startswith('kappa_g'):
            sa = label_a.replace('kappa_g', 'kappa_g_')
            sb = label_b.replace('kappa_g', 'kappa_g_')
            tick_labels.append(fr'$C^{{{sa}{sb}}}$')

        elif (label_a.startswith('g') or label_a.startswith('kappa_g')) and label_b == 'kappa_c':
            sa = label_a.replace('g', 'g_').replace('kappa_g', 'kappa_g_')
            tick_labels.append(fr'$C^{{{sa}\kappa_c}}$')

        elif (label_b.startswith('g') or label_b.startswith('kappa_g')) and label_a == 'kappa_c':
            sb = label_b.replace('g', 'g_').replace('kappa_g', 'kappa_g_')
            tick_labels.append(fr'$C^{{\kappa_c{sb}}}$')

        else:  # lens-lensing, or general case
            sa = label_a.replace('g', 'g_').replace('kappa_g', 'kappa_g_')
            sb = label_b.replace('g', 'g_').replace('kappa_g', 'kappa_g_')
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
def plot_spectra_from_dict(spectra_dict, title_prefix='Angular Power Spectrum', desired_spectra=None, plot_scaled=False, plot_linear=True, num_lens_bins = 0, num_source_bins = 0):
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
        spectra_to_plot_filtered = spectra_dict # plot all if none specified
    else:
        my_desired_pairs = create_simplified_desired_pairs(num_lens_bins, num_source_bins, desired_spectra=desired_spectra)
        for pair in my_desired_pairs:
            if pair in spectra_dict:
                spectra_to_plot_filtered[pair] = spectra_dict[pair]
            else:
                print(f"Warning: Desired spectrum {pair} not found in spectra_dict. Skipping.")

    if not spectra_to_plot_filtered:
        print("No spectra found to plot after filtering.")
        return

    # generate a colormap with enough colors for the filtered spectra
    colors = cm.get_cmap('tab10', len(spectra_to_plot_filtered))
    color_idx = 0
    plt.figure(figsize=(12, 8))
        
    for key, cl_values in spectra_to_plot_filtered.items():
        if plot_scaled:
            scaled_factor = ells * (ells + 1) / (2 * np.pi)
            y_values = cl_values * scaled_factor
            ylabel = r'$D_\ell = \ell(\ell+1)C_\ell / 2\pi$'
        else:
            y_values = cl_values
            ylabel = r'Angular Power Spectrum, $C_\ell$'

        plt.plot(ells, y_values, label=f'C_l^{{{key[0]},{key[1]}}}', color=colors(color_idx % colors.N))
        color_idx += 1

    if plot_linear:
        plt.xscale('linear')
        plt.yscale('linear')
    else:
        plt.xscale('log')
        plt.yscale('log')
    
    if plot_scaled:
        plt.title(r'Scaled Angular Power Spectra ($D_\ell$)')
    else:
        plt.title(r'Angular Power Spectra ($C_\ell$)')
            
    plt.xlabel(r'Multipole, $\ell$')
    plt.ylabel(ylabel)
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, which="both", ls="-")
    plt.tight_layout()
    plt.show()


## Covariances etc

# build data vector
def build_data_vector(forecast_map, spectra_dict, n_ell, binsize): 
    
    observed_data_vector = np.array([])
    num_binned_ells = int(np.ceil(n_ell / binsize))
    
    for pair in forecast_map.pairs:
        unbinned_cls = spectra_dict[pair]
        binned_cls_for_pair = []
        for i in range(0, n_ell, binsize):
            end_idx = min(i + binsize, len(unbinned_cls))
            if i < end_idx:
                binned_cls_for_pair.append(np.mean(unbinned_cls[i:end_idx]))
            else:
                binned_cls_for_pair.append(0.0)
        while len(binned_cls_for_pair) < num_binned_ells:
            binned_cls_for_pair.append(0.0)
        observed_data_vector = np.concatenate((observed_data_vector, binned_cls_for_pair))

    return observed_data_vector
    
# compute general spectra with noise
# z_max and n_chi are given by the examples in pyccl
#### are these fallback values ok?
def build_tracers_from_data(cosmo, lens_data, source_data, magnification_bias_lenses=None, z_max = 6, n_chi = 1024):
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

    galaxy_lensing_tracers = []
    for i in range(1, n_src + 1):
        nz = source_data[:, i]
        tracer = ccl.WeakLensingTracer(cosmo, dndz=(z_source, nz))
        galaxy_lensing_tracers.append(tracer)

    cmb_lensing_tracer = ccl.CMBLensingTracer(cosmo, z_source=1090)
    cmb_temperature_tracer = ccl.ISWTracer(cosmo, z_max = z_max, n_chi = n_chi)

    return lens_tracers, galaxy_lensing_tracers, cmb_lensing_tracer, cmb_temperature_tracer

# build tracer dictionary
def build_tracer_dict(lens_tracers, galaxy_lensing_tracers, cmb_lensing_tracer, cmb_temperature_tracer):
    tracer_dict = {'kappa_c': cmb_lensing_tracer}
    tracer_dict['T'] = cmb_temperature_tracer
    
    for i, tr in enumerate(lens_tracers):
        tracer_dict[f'g{i+1}'] = tr
    for i, tr in enumerate(galaxy_lensing_tracers):
        tracer_dict[f'kappa_g{i+1}'] = tr

    return tracer_dict

# build noise dictionary
# shot noise needs to have as many entries as lens bins
# shape noise needs to have as many entries as source bins
def build_noise_dict(f_map, ells, shot_noise_lens=None, shape_noise_source=None, cmb_noise_kk=None, cmb_noise_TT=None, cmb_noise_EE=None, cmb_noise_TE=None):
    
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
            noise_dict[(f'kappa_g{i}', f'kappa_g{i}')] = np.ones_like(ells) * shape_noise_source[i-1]

    if cmb_noise_kk is not None:
        # assuming cmb_noise_kk is already an ell-dependent array
        noise_dict[('kappa_c','kappa_c')] = cmb_noise_kk

    if cmb_noise_TT is not None:
        noise_dict[('T','T')] = cmb_noise_TT

    if cmb_noise_EE is not None:
        noise_dict[('E','E')] = cmb_noise_EE

    if cmb_noise_TE is not None:
        noise_dict[('T','E')] = cmb_noise_TE

    return noise_dict

# build spectra dictionary w/ or w/o emulator
#### CHECK
#### LEARN UNDERLYING PHYSICS
#### does this function actually consider f_map at all?
def build_spectra_dict(cosmo, f_map, tracer_dict, ells, noise_dict = None, linear_emulator = None, boost_emulator = None, cmb_primaries = False):
    spectra_dict = {}

    # make Pk2D object if an emulator is present
    if linear_emulator != None:
        a_grid = np.linspace(1/(1+5), 1.0, 20)
        z_grid = (1.0 / a_grid) - 1.0
        Pk2D_object = make_Pk2D(cosmo, linear_emulator = linear_emulator, boost_emulator = boost_emulator, z_arr = z_grid, cmin = 3.13, eta_0 = 0.60)

    # Iterate through all unique pairs of tracers in tracer_dict to calculate Cls
    # this will get all the possible pairs of (kappa_c, kappa_g, g, T)
    # I'll have to later write over TT b/c I don't think angular_cl does that properly
    tracer_labels = list(tracer_dict.keys())
    for i, label1 in enumerate(tracer_labels):
        for j, label2 in enumerate(tracer_labels):
            # ensure consistent key ordering (e.g., ('g1', 'kappa_g1') not ('kappa_g1', 'g1'))
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

    # CMB primaries -- EE, TE, TT, Ekappa_c, E_kappa_g, E_g
    # it's fine to add them all here, since in later functions we'll go in alphabetal order through the spectra_dict anyway
    if cmb_primaries:
        
        # initialize cosmology with CAMB
        h = cosmo['h']
        ombh2 = cosmo['Omega_b'] * (h**2)
        omch2 = cosmo['Omega_c'] * (h**2)
        A_s = cosmo['A_s']
        n_s = cosmo['n_s']
        Omega_k = cosmo['Omega_k']
        w0 = cosmo['w0']
        wa = cosmo['wa']
        Neff = cosmo['Neff']
        m_nu = cosmo['m_nu']
        T_CMB = cosmo['T_CMB']
        l_max = np.max(ells)

        # fix the error wherein it thinks m_nu etc is a list
        m_nu = float(m_nu[0]) if isinstance(m_nu, (list, np.ndarray)) else float(m_nu)
        Neff = float(Neff[0]) if isinstance(Neff, (list, np.ndarray)) else float(Neff)
        T_CMB = float(T_CMB[0]) if isinstance(T_CMB, (list, np.ndarray)) else float(T_CMB)

        ##### CHECK
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=h*100, ombh2=ombh2, omch2=omch2, omk=Omega_k, mnu=m_nu, nnu=Neff, TCMB=T_CMB)
        pars.InitPower.set_params(As=A_s, ns=n_s)
        pars.set_dark_energy(w=w0, wa=wa, dark_energy_model='ppf')
        pars.set_for_lmax(l_max, lens_potential_accuracy=0)

        # return raw Cl not scaled Cl -- my code generally expects the raw values
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
        lensed_cls = powers['total']  
        camb_l = np.arange(lensed_cls.shape[0])
        cmb_map = {'TT': 0, 'EE': 1, 'BB': 2, 'TE': 3}
        
        # compute primordial CMB TT, EE, TE using CAMB from the CCL cosmology parameters
        # CAMB outputs unlensed/lensed Cls up to lmax (order: TT, EE, BB, TE)
        cl_tt = np.interp(ells, camb_l, lensed_cls[:, 0])
        cl_ee = np.interp(ells, camb_l, lensed_cls[:, 1])
        cl_te = np.interp(ells, camb_l, lensed_cls[:, 3])
        spectra_dict[('T', 'T')] = cl_tt
        spectra_dict[('E', 'E')] = cl_ee
        spectra_dict[('E', 'T')] = cl_te

        # Add CMB cross-correlations with late-time tracers (kappa_c, kappa_g, g)
        # Determine active bins from tracer_dict keys
        lens_bins = [k for k in tracer_dict.keys() if k.startswith('g')]
        source_bins = [k for k in tracer_dict.keys() if k.startswith('kappa_g')]

        # E-modes do not cross-correlate with late-time structures (scalar perturbations)
        if 'kappa_c' in tracer_dict:
            spectra_dict[('E', 'kappa_c')] = np.zeros_like(ells) 
        
        for g_bin in lens_bins:
            spectra_dict[('E', g_bin)] = np.zeros_like(ells)     

        for kg_bin in source_bins:
            spectra_dict[('E', kg_bin)] = np.zeros_like(ells)    
        
    # add noise terms
    if noise_dict is not None:
        for key_noise, noise_val in noise_dict.items():
            # generally only add noise to auto-spectra.
            if key_noise[0] == key_noise[1]:
                spectra_dict[key_noise] += noise_val
            # also add TE noise (checking both orderings)
            elif (key_noise == ('E', 'T')) or (key_noise == ('T', 'E')):
                spectra_dict[('E', 'T')] += noise_val 
    
    return spectra_dict
    
# helper function to generate desired pairs based on simplified input
def create_simplified_desired_pairs(n_lens_bins, n_source_bins, desired_spectra):

    all_pairs = []

    def _canonicalize_pair(p1, p2):
        # Ensures consistent ordering, e.g., ('g1', 'kappa_c') instead of ('kappa_c', 'g1')
        # This matches the logic in ForecastMap._process_desired_pairs
        return (p1, p2) if p1 < p2 else (p2, p1)

    if 'CC' in desired_spectra:
        all_pairs.append(('kappa_c', 'kappa_c'))

    if 'GG' in desired_spectra:
        for i in range(1, n_lens_bins + 1):
            for j in range(i, n_lens_bins + 1):
                all_pairs.append(_canonicalize_pair(f'g{i}', f'g{j}'))

    if 'LL' in desired_spectra:
        for i in range(1, n_source_bins + 1):
            for j in range(i, n_source_bins + 1):
                all_pairs.append(_canonicalize_pair(f'kappa_g{i}', f'kappa_g{j}'))

    if 'GL' in desired_spectra:
        for i in range(1, n_lens_bins + 1): # lens bins first, then lensing bins for cross
            for j in range(1, n_source_bins + 1):
                all_pairs.append(_canonicalize_pair(f'g{i}', f'kappa_g{j}'))

    if 'CG' in desired_spectra: # Lens Galaxy-CMB Lensing (from Lens galaxies to CMB lensing)
        for i in range(1, n_lens_bins + 1):
            all_pairs.append(_canonicalize_pair(f'g{i}', 'kappa_c'))

    if 'CL' in desired_spectra: # CMB-Source Lensing (from CMB lensing to Source galaxies)
        for i in range(1, n_source_bins + 1):
            all_pairs.append(_canonicalize_pair(f'kappa_g{i}', 'kappa_c'))

    if 'EE' in desired_spectra:
        all_pairs.append(('E', 'E'))

    if 'TT' in desired_spectra:
        all_pairs.append(('T', 'T'))

    if 'ET' in desired_spectra or 'TE' in desired_spectra:        
        all_pairs.append(_canonicalize_pair('E', 'T'))

    if 'CE' in desired_spectra:  # CMB Lensing x CMB E-mode
        all_pairs.append(_canonicalize_pair('kappa_c', 'E'))

    if 'CT' in desired_spectra:  # CMB Lensing x CMB Temperature
        all_pairs.append(_canonicalize_pair('kappa_c', 'T'))

    if 'EL' in desired_spectra:  # CMB E-mode x Source Galaxy Lensing (Tomographic)
        for i in range(1, n_source_bins + 1):
            all_pairs.append(_canonicalize_pair('E', f'kappa_g{i}'))

    if 'LT' in desired_spectra:  # Source Galaxy Lensing x CMB Temperature (Tomographic)
        for i in range(1, n_source_bins + 1):
            all_pairs.append(_canonicalize_pair(f'kappa_g{i}', 'T'))

    if 'EG' in desired_spectra:  # CMB E-mode x Lens Galaxy Clustering (Tomographic)
        for i in range(1, n_lens_bins + 1):
            all_pairs.append(_canonicalize_pair('E', f'g{i}'))

    if 'GT' in desired_spectra:  # Lens Galaxy Clustering x CMB Temperature (Tomographic / ISW)
        for i in range(1, n_lens_bins + 1):
            all_pairs.append(_canonicalize_pair(f'g{i}', 'T'))
            
    # Remove duplicates and ensure it's a list of tuples
    return list(dict.fromkeys(all_pairs))
    
# build covariance matrix w or w/o emulator (full unless otherwise specified)
# build full matrix, potentially pass a smaller one 
#### CHECK
def build_covariance_from_data(
    cosmo,
    lens_data,
    source_data,
    f_sky,
    n_ell=3000, 
    binsize=1,  
    shot_noise_lens=None,
    shape_noise_source=None,
    cmb_noise_kk=None,
    cmb_noise_TT=None,
    cmb_noise_EE=None,
    cmb_noise_TE=None,
    magnification_bias_lenses=None, 
    desired_spectra=None,
    linear_emulator=None,
    boost_emulator=None,
    cmb_primaries=False,
    z_max=6,
    n_chi=1024
):

    full_f_map = ForecastMap(n_lens=lens_data.shape[1]-1, n_src=source_data.shape[1]-1, n_ell=n_ell, desired_pairs = None, cmb_primaries = cmb_primaries)
    
    # Use the full range of unbinned ells for CCL calculations
    ells = np.arange(2, n_ell + 2)
    
    cosmo.compute_growth()
    
    # build spectra
    lens_tracers, source_tracers, cmb_lensing_tracer, cmb_temperature_tracer = build_tracers_from_data(cosmo, lens_data, source_data, magnification_bias_lenses, z_max = z_max, n_chi = n_chi)
    tracer_dict = build_tracer_dict(lens_tracers, source_tracers, cmb_lensing_tracer, cmb_temperature_tracer)
    noise_dict = build_noise_dict(full_f_map, ells, shot_noise_lens, shape_noise_source, cmb_noise_kk, cmb_noise_TT = cmb_noise_TT, cmb_noise_EE = cmb_noise_EE, cmb_noise_TE = cmb_noise_TE)
    full_spectra_dict = build_spectra_dict(cosmo, full_f_map, tracer_dict, ells, noise_dict, linear_emulator=linear_emulator, boost_emulator=boost_emulator, cmb_primaries = cmb_primaries)

    # build covariance -- now pass the binsize to CovarianceMatrix
    full_cov = CovarianceMatrix(full_f_map, full_spectra_dict, f_sky, binsize=binsize)

    if desired_spectra is None:
        return full_cov, full_spectra_dict, full_f_map
    else:
        sliced_pairs = create_simplified_desired_pairs(lens_data.shape[1] - 1, source_data.shape[1] - 1, desired_spectra)
        sliced_f_map = ForecastMap(n_lens=lens_data.shape[1]-1, n_src=source_data.shape[1]-1, n_ell=n_ell, desired_pairs=sliced_pairs, cmb_primaries=cmb_primaries)
        # Loop over full_spectra_dict to preserve its original, chronological block order
        sliced_spectra_dict = {pair: full_spectra_dict[pair] for pair in full_spectra_dict if pair in sliced_pairs}
        sliced_cov = slice_matrix(full_cov, full_spectra_dict, full_f_map, binsize=binsize, desired_spectra=desired_spectra)
        return sliced_cov, sliced_spectra_dict, sliced_f_map

# slice vector and matrix given desired pairs
##### Modify to make the return sliced matrix a real CovarianceMatrix object??
def slice_matrix(
    cov_obj, 
    spectra_dict, 
    f_map, 
    binsize=1, 
    desired_spectra=None
):
    
    if desired_spectra is None:
        pairs_to_slice = f_map.pairs
    else:
        desired_pairs = create_simplified_desired_pairs(
            n_lens_bins=f_map.n_lens, 
            n_source_bins=f_map.n_src, 
            desired_spectra=desired_spectra
        )
        
        processed_desired_pairs = []
        for p in f_map.pairs:
            if p in desired_pairs:
                if not isinstance(p, tuple) or len(p) != 2:
                    raise ValueError(f"Each desired pair must be a tuple of two strings: {p}")
                
                # Canonical ordering (e.g., matching ('kappa_c', 'g') instead of ('g', 'kappa_c'))
                if p[0] > p[1]:
                    canonical_pair = (p[1], p[0])
                else:
                    canonical_pair = p
                    
                if canonical_pair not in processed_desired_pairs:
                    processed_desired_pairs.append(canonical_pair)
                    
            pairs_to_slice = processed_desired_pairs
        
    # Collect the global index ranges using f_map.get_indices
    all_ranges = []
    final_sliced_pairs = []

    for pair in f_map.pairs:
        if pair in pairs_to_slice:
            try:
                # Let ForecastMap find the start and end indices for this block
                start, end = f_map.get_indices(pair)
                start = int(start / binsize)
                end = int(end / binsize)
                all_ranges.append(np.arange(start, end))
                final_sliced_pairs.append(pair)
            except ValueError as e:
                # Skip any blocks that don't exist in the current global configuration
                print(f"Warning: {e} Skipping this block from the slice.")
                continue

    if not all_ranges:
        raise ValueError("No matching spectra blocks were found to slice!")

    # Concatenate all index segments into a single array
    keep_indices = np.concatenate(all_ranges)
    
    # Extract the underlying raw matrix from the input object if needed
    # This handles both raw numpy arrays and object wrappers gracefully
    full_matrix = cov_obj.matrix if hasattr(cov_obj, 'matrix') else cov_obj

    # Double-axis slicing to extract sub-blocks
    sliced_matrix_raw = full_matrix[keep_indices, :]
    sliced_matrix_raw = sliced_matrix_raw[:, keep_indices]
    
    # --- FIX: Wrap the matrix in a class container to preserve properties ---
    ### FIX CAUSE THIS IS NOW AN IMPROPER OBJECT -- THE MATRIX IS CUT CORRECTLY BUT THE OTHER STUFF WILL BE WRONG
    from copy import copy
    sliced_cov_obj = copy(cov_obj)
    sliced_cov_obj.matrix = sliced_matrix_raw
    
    return sliced_cov_obj

# get parameter dict from a given cosmology
def extract_param_dict(cosmology):

    h = cosmology['h']
    Omega_b = cosmology['Omega_b']
    Omega_c = cosmology['Omega_c']
    A_s = cosmology['A_s']
    n_s = cosmology['n_s']
    w0 = cosmology['w0']
    wa = cosmology['wa']
    Omega_k = cosmology['Omega_k']  
    Neff = cosmology['Neff']
    m_nu = cosmology['m_nu']
    T_CMB = cosmology['T_CMB']
    
    # Reconstruct Omega_m and Omega_lambda from cold dark matter and baryons
    Omega_m = Omega_c + Omega_b
    Omega_lambda = 1 - Omega_m - Omega_k
    
    fiducial_params = {
        'Omega_m': Omega_m,
        'Omega_b': Omega_b,
        'Omega_lambda': Omega_lambda,
        'h':       h,
        'A_s':     A_s,
        'n_s':     n_s,
        'w0':      w0,
        'wa':      wa,
        'Omega_k': Omega_k,
        'Neff': Neff,
        'm_nu': m_nu,
        'T_CMB': T_CMB,
    }
    
    return fiducial_params
        
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
#### THIS DOES NOT ACCOUNT FOR DARK ENERGY, ETC
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
# the order of forecastmaps should be consistent across the board
class ForecastMap:
    def __init__(self, n_lens=4, n_src=4, n_ell=20, desired_pairs=None, cmb_primaries=False):
        self.n_lens = n_lens
        self.n_src = n_src
        self.n_ell = n_ell
        self.cmb_primaries = cmb_primaries
        
        # list of unique spectra
        if desired_pairs:
            self.pairs = self._process_desired_pairs(desired_pairs)
        else:
            self.pairs = self._generate_all_pairs()

        self.pair_to_index = {p: i for i, p in enumerate(self.pairs)}

        # length of data vector
        self.vector_length = len(self.pairs) * n_ell

    def _generate_all_pairs(self): 
        p = []

        # CMB lensing convergence auto spectrum
        p += [('kappa_c','kappa_c')]

        # Lens galaxies auto/cross -- not zero indexing because of convention
        for i in range(1, self.n_lens + 1):
            for j in range(i, self.n_lens + 1):
                p.append((f'g{i}', f'g{j}'))

        # Galaxy lensing auto/cross
        for i in range(1, self.n_src + 1):
            for j in range(i, self.n_src + 1):
                p.append((f'kappa_g{i}', f'kappa_g{j}'))

        # Lens galaxies-galaxy lensingcross
        for i in range(1, self.n_lens + 1):
            for j in range(1, self.n_src + 1):
                p.append((f'g{i}', f'kappa_g{j}'))

        # Lens galaxies -- CMB lensing cross
        for i in range(1, self.n_lens + 1):
            p.append((f'g{i}', f'kappa_c'))

        # Galaxy lensing -- CMB lensing cross
        for j in range(1, self.n_src + 1):
            p.append((f'kappa_g{j}', f'kappa_c'))
            
        # add cmb_primaries, if desired
        if self.cmb_primaries:
            # CMB temperature auto-spectrum
            p += [('T', 'T')]
    
            # CMB E-mode auto-spectrum
            p += [('E', 'E')]
    
            # CMB temperature-E-mode cross-spectra
            p += [('E', 'T')]

            # CMB primary -- CMB lensing
            p += [('T', 'kappa_c')]
            p += [('E', 'kappa_c')]

            # CMB primary -- lens galaxies cross
            for i in range(1, self.n_lens + 1):
                p.append((f'g{i}', 'T'))
            for i in range(1, self.n_lens + 1):
                p.append((f'g{i}', 'E'))
 
            # CMB primary -- galaxy lensing cross
            for j in range(1, self.n_src + 1):
                p.append((f'kappa_g{j}', 'T'))
            for j in range(1, self.n_src + 1):
                p.append((f'kappa_g{j}', 'E'))
                
        # Canonicalize every single pair internally, then sort alphabetically
        canonical_pairs = []
        for pair in p:
            if pair[0] > pair[1]:
                canonical_pairs.append((pair[1], pair[0]))
            else:
                canonical_pairs.append(pair)
                
        # Sort the list of tuples lexicographically
        canonical_pairs.sort()
        
        return canonical_pairs

    def _process_desired_pairs(self, desired_pairs_input):
        # Ensure consistent ordering and uniqueness for desired_pairs
        user_pairs = []
        for pair in desired_pairs_input:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(f"Each desired pair must be a tuple of two strings: {pair}")

            # Canonical ordering: ensure the first element is lexicographically smaller
            # This aligns with how build_spectra_dict stores Cls (label1 < label2)
            if pair[0] > pair[1]:
                canonical_pair = (pair[1], pair[0])
            else:
                canonical_pair = pair

            if canonical_pair not in user_pairs:
                user_pairs.append(canonical_pair)
                
        # make sure order is the same as it is when generate_all_pairs is used
        user_pairs_set = list(dict.fromkeys(user_pairs))
        master_order = self._generate_all_pairs()
        processed_pairs = [pair for pair in master_order if pair in user_pairs_set]
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
                print("cross-spectrum for ", x, ", ", y, " is not given, and is assumed to be zero.")
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
#### Fix if desired
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

# -------------------------------------------------------------------------------------------------------------------------------------------- #

## LIKELIHOODS 

# SO DESI Likelihood (w/ or w/o emulator, w or w/o primaries) 
class SO_x_DESI_Likelihood(Likelihood):

    params = {
        "Omega_m": None, # matter density
        "A_s": None,     # amplitude of primordial fluctuations
        "h": None,       # Hubble parameter
        "Omega_b": None, # baryon density
        "n_s": None,     # primordial tilt
        "w0": None,      # dark energy equation of state parameter
        "wa": None,      # dark energy equation of state parameter evolution
        "Omega_k": None, # curvature density (for curved LCDM) - will set to 0 for flat_LCDM
        "Neff": None,     # effective number of massless neutrinos present -- defaults to 3.044
        "m_nu": None,     # eass in eV of the massive neutrinos present
        "T_CMB": None    # contemporary tempature of the CMB
    }

    # data-related settings, to be defined when configuring Cobaya
    data_specs: dict

    # initialize the likelihood
    # set up fiducial cosmology, calculate fiducial data vector and covariance matrix
    def initialize(self):
        print("Initializing SO_x_DESI_Likelihood...")

        # get emulators
        self.emulator = self.data_specs.get('emulator', False)
        if self.emulator:
            print("Using emulator.")
            self.linear_emulator = CPJ(probe='mpk_lin')
            self.boost_emulator = CPJ(probe='mpk_boost')
        else:
            print("Not using emulator.")
            self.linear_emulator = None
            self.boost_emulator = None
        
        # extract necessary data specifications from the Cobaya input YAML/dictionary
        self.f_sky = self.data_specs.get('f_sky', 0.4) # default to 0.4 if not provided
        self.n_ell = self.data_specs.get('n_ell', 3000) # max unbinned ell
        self.binsize = self.data_specs.get('binsize', 50) # binning size for ell
        self.magnification_bias_lenses = self.data_specs.get('magnification_bias_lenses', 0.8)
        self.z_max = self.data_specs.get('z_max', 6)
        self.n_chi = self.data_specs.get('n_chi', 1024)
        self.cmb_primaries = self.data_specs.get('cmb_primaries')

        # set desired spectra
        desired_spectra = self.data_specs.get('desired_spectra')
        if desired_spectra != 'None':
            self.desired_spectra = desired_spectra
        elif self.cmb_primaries:
            self.desired_spectra = ['GG', 'LL', 'GL', 'CC', 'CL', 'CG', 'TT', 'EE', 'ET', 'GT', 'LT', 'CT', 'EG', 'EL', 'CE']
        else:
            self.desired_spectra = ['GG', 'LL', 'GL', 'CC', 'CL', 'CG']
        print("Spectra considered: ", self.desired_spectra)
        
        # noise parameters, loaded from files, with default 'None'
        shot_noise_path = self.data_specs.get('shot_noise_path')
        if shot_noise_path:
            print(f"  Loading lens shot noise from: {shot_noise_path}")
            self.shot_noise_lens = np.load(shot_noise_path)
        else:
            self.shot_noise_lens = None

        shape_noise_path = self.data_specs.get('shape_noise_path')
        if shape_noise_path:
            print(f"  Loading source shape noise from: {shape_noise_path}")
            self.shape_noise_source = np.load(shape_noise_path)
        else:
            self.shape_noise_source = None

        cmb_noise_kk_path = self.data_specs.get('cmb_noise_kk_path')
        if cmb_noise_kk_path:
            print(f"  Loading CMB noise from: {cmb_noise_kk_path}")
            self.cmb_noise_kk = np.load(cmb_noise_kk_path)
        else:
            self.cmb_noise_kk = None

        cmb_noise_TT_path = self.data_specs.get('cmb_noise_TT_path')
        if cmb_noise_TT_path:
            print(f"  Loading CMB noise from: {cmb_noise_TT_path}")
            self.cmb_noise_TT = np.load(cmb_noise_TT_path)
        else:
            self.cmb_noise_TT = None
        
        cmb_noise_EE_path = self.data_specs.get('cmb_noise_EE_path')
        if cmb_noise_EE_path:
            print(f"  Loading CMB noise from: {cmb_noise_EE_path}")
            self.cmb_noise_EE = np.load(cmb_noise_EE_path)
        else:
            self.cmb_noise_EE = None
            
        cmb_noise_TE_path = self.data_specs.get('cmb_noise_TE_path')
        if cmb_noise_TE_path:
            print(f"  Loading CMB noise from: {cmb_noise_TE_path}")
            self.cmb_noise_TE = np.load(cmb_noise_TE_path)
        else:
            self.cmb_noise_TE = None

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
            desired_pairs = create_simplified_desired_pairs(num_lens_bins, num_source_bins, self.desired_spectra)
            self.f_map = ForecastMap(n_lens=num_lens_bins, n_src=num_source_bins, n_ell=self.n_ell,
                                     desired_pairs=desired_pairs, cmb_primaries = self.cmb_primaries)

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

            _Omega_c = fiducial_cosmo_input.get('Omega_c')
            _Omega_b = fiducial_cosmo_input.get('Omega_b')
            _h = fiducial_cosmo_input.get('h')
            _A_s = fiducial_cosmo_input.get('A_s')
            _n_s = fiducial_cosmo_input.get('n_s')
            _w0 = fiducial_cosmo_input.get('w0')
            _wa = fiducial_cosmo_input.get('wa')
            _Omega_k = fiducial_cosmo_input.get('Omega_k')
            _Neff = fiducial_cosmo_input.get('Neff')
            _m_nu = fiducial_cosmo_input.get('m_nu')
            _T_CMB = fiducial_cosmo_input.get('T_CMB')

            self.fiducial_cosmology = ccl.Cosmology(
                Omega_c=_Omega_c,
                Omega_b=_Omega_b,
                h=_h,
                A_s=_A_s,
                n_s=_n_s,
                w0=_w0,
                wa=_wa,
                Omega_k=_Omega_k,
                Neff=_Neff,
                m_nu=_m_nu,
                T_CMB=_T_CMB,
                transfer_function='boltzmann_camb',
                extra_parameters={"camb": {"dark_energy_model": "ppf"}}
            )
            print(f"  Fiducial Cosmology parameters: Omega_c={_Omega_c}, Omega_b={_Omega_b}, h={_h}, A_s={_A_s}, n_s={_n_s}, w0={_w0}, wa={_wa}, Omega_k={_Omega_k}, Neff = {_Neff}, m_nu={_m_nu}, T_CMB={_T_CMB}")

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
                    cmb_noise_kk=self.cmb_noise_kk,
                    cmb_noise_TT=self.cmb_noise_TT,
                    cmb_noise_EE=self.cmb_noise_EE,
                    cmb_noise_TE=self.cmb_noise_TE,
                    magnification_bias_lenses=self.magnification_bias_lenses,
                    desired_spectra=self.desired_spectra,
                    linear_emulator=self.linear_emulator,
                    boost_emulator=self.boost_emulator,
                    cmb_primaries=self.cmb_primaries,
                    z_max=self.z_max,
                    n_chi=self.n_chi
                )
            self.covariance_matrix = self.cov_obj.matrix 

            self.observed_data_vector = build_data_vector(self.f_map, self.fiducial_spectra_dict, self.n_ell, self.binsize)

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
        Omega_b = kwargs['Omega_b']
        h = kwargs['h']
        A_s = kwargs['A_s']
        n_s = kwargs['n_s']
        w0 = kwargs['w0']
        wa = kwargs['wa']
        Omega_k = kwargs['Omega_k']
        Neff = kwargs['Neff']
        m_nu = kwargs['m_nu']
        T_CMB = kwargs['T_CMB']
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
            Neff=Neff,
            m_nu=m_nu,
            T_CMB=T_CMB,
            transfer_function='boltzmann_camb',
            extra_parameters={"camb": {"dark_energy_model": "ppf"}}
        )
        
        current_cosmology.compute_growth()
        
        # calculate the theoretical model data vector M(theta) for the current cosmology
        ells = np.arange(2, self.n_ell + 2) # unbinned ells
        lens_tracers, source_tracers, cmb_lensing_tracer, cmb_temperature_tracer = build_tracers_from_data(
            current_cosmology, self.lens_data, self.source_data, self.magnification_bias_lenses)
        tracer_dict = build_tracer_dict(lens_tracers, source_tracers, cmb_lensing_tracer, cmb_temperature_tracer)
        noise_dict = build_noise_dict(self.f_map, ells, self.shot_noise_lens, self.shape_noise_source, cmb_noise_kk = self.cmb_noise_kk, cmb_noise_TT = self.cmb_noise_TT, cmb_noise_EE = self.cmb_noise_EE, cmb_noise_TE = self.cmb_noise_TE)
        current_spectra_dict = build_spectra_dict(current_cosmology, self.f_map, tracer_dict, ells, noise_dict, linear_emulator=self.linear_emulator, boost_emulator=self.boost_emulator, cmb_primaries = self.cmb_primaries)

        # flatten the current Cls into a model data vector 'M'
        model_data_vector = build_data_vector(self.f_map, current_spectra_dict, self.n_ell, self.binsize)

        # calculate the difference vector (D - M(theta))
        difference_vector = self.observed_data_vector - model_data_vector

        # calculate the log-likelihood
        # ln L = -1/2 * (D - M)^T * C^-1 * (D - M) - 1/2 * ln|C|
        chi2 = difference_vector.dot(self.inv_covariance.dot(difference_vector))
        log_likelihood = -0.5 * chi2 - 0.5 * self.log_det_covariance

        return log_likelihood
    
    def profile_chi2(self, **kwargs):
        Omega_m = kwargs.get('Omega_m')
        Omega_b = kwargs.get('Omega_b')
        Omega_c = Omega_m - Omega_b
        h = kwargs.get('h')
        A_s = kwargs.get('A_s')
        n_s = kwargs.get('n_s')
        w0 = kwargs.get('w0')
        wa = kwargs.get('wa')
        Omega_k = kwargs.get('Omega_k')
        Neff = kwargs.get('Neff')
        m_nu = kwargs.get('m_nu')
        T_CMB = kwargs.get('T_CMB')
        
        # Initialize the cosmology using CCL
        current_cosmology = ccl.Cosmology(
            Omega_c=Omega_c, Omega_b=Omega_b, h=h, A_s=A_s, 
            n_s=n_s, w0=w0, wa=wa, Omega_k=Omega_k, Neff=Neff, m_nu=m_nu, T_CMB=T_CMB,
            transfer_function='boltzmann_camb',
            extra_parameters={"camb": {"dark_energy_model": "ppf"}}
        )
        current_cosmology.compute_growth()
        
        # Build tracers and noise dictionaries
        ells = np.arange(2, self.n_ell + 2)
        lens_tracers, source_tracers, cmb_lensing_tracer, cmb_temperature_tracer = build_tracers_from_data(
            current_cosmology, self.lens_data, self.source_data, self.magnification_bias_lenses)
        tracer_dict = build_tracer_dict(lens_tracers, source_tracers, cmb_lensing_tracer, cmb_temperature_tracer)
        noise_dict = build_noise_dict(self.f_map, ells, self.shot_noise_lens, self.shape_noise_source, cmb_noise_kk = self.cmb_noise_kk, cmb_noise_TT = self.cmb_noise_TT, cmb_noise_EE = self.cmb_noise_EE, cmb_noise_TE = self.cmb_noise_TE)
        current_spectra_dict = build_spectra_dict(current_cosmology, self.f_map, tracer_dict, ells, noise_dict, linear_emulator=self.linear_emulator, boost_emulator=self.boost_emulator, cmb_primaries = self.cmb_primaries)

        # flatten the current Cls into a model data vector 'M'
        model_data_vector = build_data_vector(self.f_map, current_spectra_dict, self.n_ell, self.binsize)

        # Calculate the residual vector (D - M)
        difference_vector = self.observed_data_vector - model_data_vector
        
        # Calculate Chi-squared: r^T * InvCov * r
        chi2 = difference_vector.dot(self.inv_covariance.dot(difference_vector))
        
        return chi2
        
# -------------------------------------------------------------------------------------------------------------------------------------------- #

# Fisher Forecast class
class FisherForecaster:
    def __init__(self, cosmology, lens_data, source_data, f_sky=0.4, n_ell=5000, binsize=50, 
                 shot_noise_lens=None, shape_noise_source=None, cmb_noise_kk=None, cmb_noise_TT=None, 
                 cmb_noise_EE=None, cmb_noise_TE=None, magnification_bias_lenses=None, desired_spectra=None, 
                 linear_emulator=None, boost_emulator=None, step_dict=None, cmb_primaries=False, z_max=6, n_chi=1024):

        self.cosmology = cosmology
        self.lens_data = lens_data
        self.source_data = source_data
        self.cmb_primaries = cmb_primaries
        self.z_max = z_max
        self.n_chi = n_chi
    
        self.survey_params = {
            'f_sky': f_sky, 'n_ell': n_ell, 'binsize': binsize,
            'shot_noise_lens': shot_noise_lens, 'shape_noise_source': shape_noise_source,
            'cmb_noise_kk': cmb_noise_kk, 'cmb_noise_TT': cmb_noise_TT, 'cmb_noise_EE': cmb_noise_EE, 'cmb_noise_TE': cmb_noise_TE,
            'magnification_bias_lenses': magnification_bias_lenses, 'desired_spectra': desired_spectra, 
            'linear_emulator': linear_emulator, 'boost_emulator': boost_emulator, 'cmb_primaries': cmb_primaries, 
            'z_max': z_max, 'n_chi': n_chi
        }
        
        # step sizes for numerical derivatives
        self.step_dict = step_dict if step_dict is not None else {
            'Omega_m': 1e-2, 'A_s': 2e-11, 'h': 1e-3, 'w0': 1e-2, 'wa': 1e-2, 'n_s': 1e-3, 'Omega_b': 1e-4, 
            'Omega_c': 1e-4, 'Omega_k': 1e-3, 'Neff': 1e-3, 'm_nu': 1e-3, 'T_CMB': 1e-3, 'Omega_lambda': 1e-3 
            ### consider changing the last four, I put these as temp values
        }
        
        # extract and freeze our baseline fiducial truths
        self.fiducial_dict = self._extract_param_dict(self.cosmology)
        
        # matrices initialized to None until computed
        self.F = None    # Fisher matrix
        self.cov = None  # parameter covariance matrix

        # build full f_map, etc. so that I don't have to remake them every time I call build_theory_vector
        p = self.survey_params
        self.ells = np.arange(2, p['n_ell'] + 2) 
        ##### FINISH ADDING l_min, make sure FMap and CovMatrix can handle it
        ##### ADD TO MCMC TOO
        self.num_binned_ells = int(np.ceil(p['n_ell'] / p['binsize']))
        self.full_f_map = ForecastMap(n_lens=self.lens_data.shape[1]-1, n_src=self.source_data.shape[1]-1, n_ell=p['n_ell'], cmb_primaries = self.cmb_primaries)

        # default to full spectra
        if p['desired_spectra'] is None: 
            if self.cmb_primaries:
                p['desired_spectra'] = ['GG', 'LL', 'GL', 'CC', 'CL', 'CG', 'TT', 'EE', 'ET', 'GT', 'LT', 'CT', 'EG', 'EL', 'CE']
            else: 
                p['desired_spectra'] = ['GG', 'LL', 'GL', 'CC', 'CL', 'CG']

        sliced_pairs = create_simplified_desired_pairs(self.lens_data.shape[1] - 1, self.source_data.shape[1] - 1, p['desired_spectra'])
        self.final_f_map = ForecastMap(n_lens=self.lens_data.shape[1]-1, n_src=self.source_data.shape[1]-1, n_ell=p['n_ell'], desired_pairs=sliced_pairs, cmb_primaries = self.cmb_primaries)
        self.sliced_pairs = sliced_pairs
        
    # get parameter dictionary from a ccl cosmology (I usually pass cosmologies, not dictionaries)
    def _extract_param_dict(self, cosmology):
        h = cosmology['h']
        Omega_b = cosmology['Omega_b']
        Omega_c = cosmology['Omega_c']
        Omega_m = cosmology['Omega_m']
        A_s = cosmology['A_s']
        n_s = cosmology['n_s']
        w0 = cosmology['w0']
        wa = cosmology['wa']
        Omega_k = cosmology['Omega_k']
        Neff = cosmology['Neff']
        m_nu = cosmology['m_nu']
        T_CMB = cosmology['T_CMB']
        
        return {
            'Omega_m': Omega_m,
            'Omega_b': Omega_b,
            'Omega_c': Omega_c,
            'Omega_k': Omega_k,
            'Omega_lambda': 1 - Omega_m - Omega_k,
            'h':       h,
            'A_s':     A_s,
            'n_s':     n_s,
            'w0':      w0,
            'wa':      wa,
            'Neff':    Neff,
            'm_nu':    m_nu,
            'T_CMB':   T_CMB
        }

    # build a data vector given parameters
    def build_theory_vector(self, cosmology, silent=True):

        if not silent:
            print("Computing theory vector...")
    
        cosmology.compute_growth()
        p = self.survey_params
        
        lens_tracers, source_tracers, cmb_lensing_tracer, cmb_temperature_tracer = build_tracers_from_data(
            cosmology, self.lens_data, self.source_data, p['magnification_bias_lenses'], z_max = p['z_max'], n_chi=p['n_chi'])
        tracer_dict = build_tracer_dict(lens_tracers, source_tracers, cmb_lensing_tracer, cmb_temperature_tracer)
        noise_dict = build_noise_dict(self.full_f_map, self.ells, p['shot_noise_lens'], p['shape_noise_source'], cmb_noise_kk = p['cmb_noise_kk'], cmb_noise_TT = p['cmb_noise_TT'], cmb_noise_EE = p['cmb_noise_EE'], cmb_noise_TE = p['cmb_noise_TE'])
        full_spectra_dict = build_spectra_dict(cosmology, self.full_f_map, tracer_dict, self.ells, noise_dict, linear_emulator=p['linear_emulator'], boost_emulator=p['boost_emulator'], cmb_primaries=self.cmb_primaries)
        
        if self.sliced_pairs is not None: 
            final_spectra_dict = {pair: full_spectra_dict[pair] for pair in full_spectra_dict if pair in self.sliced_pairs}
        else:
            final_spectra_dict = full_spectra_dict

        self.spectra_dict = final_spectra_dict
        
        model_data_vector = []
        for pair in self.final_f_map.pairs:
            unbinned_cls = final_spectra_dict[pair]
            for i in range(0, p['n_ell'], p['binsize']):
                end_idx = min(i + p['binsize'], len(unbinned_cls))
                if i < end_idx:
                    model_data_vector.append(np.mean(unbinned_cls[i:end_idx]))
                else:
                    model_data_vector.append(0.0)
        
        return np.array(model_data_vector)

    # manually compute derivatives
    def get_derivatives(self, desired_params):
        C_derivatives = {}
        mu_derivatives = {}
        p = self.survey_params
        
        for param in desired_params:
            # get step size, default to 1e-3
            step = self.step_dict.get(param, 1e-3)
            
            params_up = self.fiducial_dict.copy()
            params_down = self.fiducial_dict.copy()
            params_up[param] += step
            params_down[param] -= step

            # since I'm generally sampling over Omega_m, not Omega_c
            # when taking derivatives and varying Omega_b I'll keep Omega_m constant by adjusting Omega_c under the hood
            #### THIS IS A PROBLEM -- omega_m NEEDS TO BE ABLE TO VARY FREELY AND NOT CORRESPOND SIMPLY TO omega_c
            ### RIGHT NOW VARYING OMEGA_M JUST MEANS VARYING OMEGA_C
            cosmology_up = ccl.Cosmology(
                Omega_c = params_up['Omega_c'],
                Omega_b = params_up['Omega_b'],
                Omega_k = params_up['Omega_k'],
                h       = params_up['h'],
                A_s     = params_up['A_s'],
                n_s     = params_up['n_s'],
                w0      = params_up['w0'],
                wa      = params_up['wa'],
                Neff    = params_up['Neff'],
                m_nu    = params_up['m_nu'],
                T_CMB   = params_up['T_CMB'],
                transfer_function = 'boltzmann_camb',
                extra_parameters={"camb": {"dark_energy_model": "ppf"}}
            )
            
            cosmology_down = ccl.Cosmology(
                Omega_c = params_down['Omega_c'],
                Omega_b = params_down['Omega_b'],
                Omega_k = params_down['Omega_k'],
                h       = params_down['h'],
                A_s     = params_down['A_s'],
                n_s     = params_down['n_s'],
                w0      = params_down['w0'],
                wa      = params_down['wa'],
                Neff    = params_down['Neff'],
                m_nu    = params_down['m_nu'],
                T_CMB   = params_down['T_CMB'],
                transfer_function = 'boltzmann_camb',
                extra_parameters={"camb": {"dark_energy_model": "ppf"}}
            )
            
            cosmology_up.compute_growth()
            cosmology_down.compute_growth()

            mu_up = self.build_theory_vector(cosmology_up)
            mu_down = self.build_theory_vector(cosmology_down)
            mu_derivatives[param] = (mu_up - mu_down) / (2.0 * step)

            cov_obj_up, _, _ = build_covariance_from_data(cosmology_up, self.lens_data, self.source_data, **p)
            cov_obj_down, _, _ = build_covariance_from_data(cosmology_down, self.lens_data, self.source_data, **p)
            C_derivatives[param] = (cov_obj_up.matrix - cov_obj_down.matrix) / (2.0 * step)
            
        return C_derivatives, mu_derivatives

    # build and invert Fisher matrix without considering priors, since they are uniform, not Gaussian
    def make_fisher_matrix(self, desired_params=None, C=None, mu=None, C_derivatives=None, mu_derivatives=None, print_summary = False):
        
        # default to full Fisher matrix
        if desired_params is None:
            desired_params = ['Omega_c', 'A_s', 'h', 'w0', 'wa', 'n_s', 'Omega_b', 'Omega_k', 'Neff', 'm_nu', 'T_CMB']
        
        self.desired_params = desired_params
        p = self.survey_params

        # build fiducial covariance and theory vector if they are missing
        if C is None:
            cov_obj, _, _ = build_covariance_from_data(self.cosmology, self.lens_data, self.source_data, **p)
            C = cov_obj.matrix
        if mu is None:
            mu = self.build_theory_vector(self.cosmology)
            
        if C_derivatives is None or mu_derivatives is None:
            C_derivatives, mu_derivatives = self.get_derivatives(desired_params)

        n_params = len(desired_params)
        F = np.zeros((n_params, n_params))
        inv_C = np.linalg.inv(C)
        
        for i, p_i in enumerate(desired_params):
            for j, p_j in enumerate(desired_params):
                dC_di = C_derivatives[p_i]
                dC_dj = C_derivatives[p_j]
                
                dmu_di = mu_derivatives[p_i][:, np.newaxis]
                dmu_dj = mu_derivatives[p_j][:, np.newaxis]

                matrix1 = inv_C @ dC_di @ inv_C @ dC_dj
                matrix2 = inv_C @ ((dmu_di @ dmu_dj.T) + (dmu_dj @ dmu_di.T))
                F[i, j] = 0.5 * np.trace(matrix1 + matrix2)
        
        self.F = F
        self.cov = np.linalg.inv(F)   
        
        # print results in a highly readable format
        if print_summary:
            print("")
            print("Fisher Forecast Results Without Priors")
            for i, p_name in enumerate(self.desired_params):
                sigma = np.sqrt(self.cov[i, i])
                error_str = f"{sigma:.3e}" if sigma < 0.001 else f"{sigma:.4f}"
                print(f"The uncertainty on {p_name} is {error_str}")
            print("")
        
        return self.F, self.cov

    # compute partial derivative between paramaters for Jacobian
    # derivative of param1 wrt param2
    def get_partial_derivative(self, parameter1, parameter2):

        param1 = parameter1.lower()
        param2 = parameter2.lower()
        
        if param1 == param2:
            return 1

        elif param2 == 'omega_m':
            if param1 in ['omega_c', 'omega_b']:
                return 1
                
        elif param2 == 'omega_lambda':
            if param1 in ['omega_c', 'omega_b', 'omega_k']:
                return -1
                
        else:
            return 0
            
    # project matrix from one set of parameters to another
    # e.g. omega_b omega_c omega_k to omega_b omega_m omega_k
    def project_fisher_matrix(self, F_raw, sampled_params, plot_params):

        print("WARNING: Our get_partial_derivative function is only equipped to handle omega_m, omega_b, omega_c, omega_k, h, A_s, n_s, Neff, m_nu, and T_CMB. If you have passed over parameters to it DO NOT TRUST THE RESULTS.")
        
        # if the bases are identical, no projection is needed
        if list(sampled_params) == list(plot_params):
            return F_raw

        # build Jacobian matrix 
        n_sampled = len(sampled_params)
        n_plotted = len(plot_params)
        J = np.zeros((n_sampled, n_plotted))
        
        for i in range(n_sampled):
            for j in range(n_plotted):
                # each entry is the partial derivative of the old parameter wrt the new parameter
                J[i][j] = self.get_partial_derivative(sampled_params[i], plot_params[j])
                
        # Transform: F_new = J^T @ F_raw @ J
        F_projected = J.T @ F_raw @ J
    
        return F_projected

    def sample_fisher_with_uniform_priors(self, uniform_priors=None, num_samples=200000):
        
        if self.cov is None:
            raise ValueError("Covariance matrix 'self.cov' is not computed yet. Run your Fisher execution pipeline first.")
            
        # gather the center (fiducial values) in the exact order of desired_params
        fiducial_values = [self.fiducial_dict[p] for p in self.desired_params]
        
        # draw rapid multivariate normal samples based on the Fisher covariance
        samples = np.random.multivariate_normal(fiducial_values, self.cov, size=num_samples)
        
        # apply uniform prior cuts (mask out samples that exceed boundaries)
        if uniform_priors is not None:
            mask = np.ones(num_samples, dtype=bool)
            for idx, param_name in enumerate(self.desired_params):
                if param_name in uniform_priors:
                    p_min, p_max = uniform_priors[param_name]
                    # Update mask to only keep samples within the hard walls
                    mask &= (samples[:, idx] >= p_min) & (samples[:, idx] <= p_max)
            
            samples = samples[mask]
            
            if len(samples) == 0:
                raise ValueError("Zero samples survived the uniform prior cuts. Check if your fiducial values sit outside your prior bounds!")

        # Convert into a GetDist MCSamples object for seamless plotting
        param_labels = [p for p in self.desired_params] 

        if uniform_priors is None:
            name_tag = "Fisher (without priors)"
        else:
            name_tag = "Fisher (with uniform priors)"
            
        mcsamples = MCSamples(
            samples=samples, 
            names=self.desired_params, 
            labels=param_labels, 
            name_tag=name_tag
        )
        
        return mcsamples

    # generate contour plot with or w/o Cobaya and prior-less version overlaid
    # for prior-less forecasts, make simple Gaussian ellipses instead of sampling
    # if we want to plot paramaters that were not sampled over -- e.g. omega_m we'll need to convert
    def plot_with_cobaya_overlay(
        self, 
        param_names=None, 
        title="Fisher Forecast vs. Cobaya MCMC Constraints",
        cobaya_chain_dir=None, 
        sampled_params=None, # the parameters sampled over, e.g. omega_k, omega_b, omega_c
        plot_params=None,    # the parameters plotted, e.g. omega_k, omega_b, omega_m
        num_chains=4, 
        burn_in_fraction=0.2,
        uniform_priors=None,
        overlay_priorless=False, 
        num_samples=200000,
        save_plot = False,
        plot_folder = "plots/Fisher Forecasts",
        print_summary = False,
        plot_prior = False
    ):

        # Default fallback if parameters aren't provided
        if sampled_params is None:
            sampled_params = self.desired_params
        if plot_params is None:
            plot_params = sampled_params
            
        raw_datasets = []
        legend_labels = []
        contour_colors = []

        latex_labels = {'Omega_m': r'\Omega_\mathrm{m}',
                        'Omega_b': r'\Omega_\mathrm{b}',
                        'Omega_k': r'\Omega_\mathrm{k}',
                        'Omeag_c': r'\Omega_\mathrm{c}',
                        'wa': r'w_a',
                        'w0': r'w_0',
                        'h': r'h',
                        'A_s': r'A_\mathrm{s}',
                        'n_s': r'n_\mathrm{s}',
                        'Neff': r'N_\mathrm{eff}',
                        'm_nu': r'm_\mathrm{nu}',
                        'T_CMB': r'T_\mathrm{CMB}'
                    }
        labels = [latex_labels.get(p, p) for p in param_names]

        #### test -- concerned about slicing 
        # Helper to extract the correct sliced covariance matrix for GetDist
        def get_sliced_cov_and_fiducial():
            fiducial_values = [float(self.fiducial_dict[p]) for p in param_names]
            # Ensure we pull indices corresponding to the selected param_names
            indices = [self.desired_params.index(p) for p in param_names]
            sliced_cov = self.cov[np.ix_(indices, indices)]
            return fiducial_values, sliced_cov

        # Get main Fisher results
        fisher_dataset = self.sample_fisher_with_uniform_priors(uniform_priors=uniform_priors, num_samples=num_samples)
        raw_datasets.append(fisher_dataset)
        legend_labels.append("Fisher Forecast (With Priors)" if uniform_priors else "Fisher Forecast")
        contour_colors.append("firebrick")

        # Optional: overlay the prior-less Fisher distribution
        if overlay_priorless and uniform_priors is not None:
            fiducial_values, sliced_cov = get_sliced_cov_and_fiducial()
            ### GAUSSIANND OBJECT -- CHANGE
            fisher_priorless = GaussianND(fiducial_values, sliced_cov, names=param_names, labels=labels)
            raw_datasets.append(fisher_priorless)
            legend_labels.append("Fisher Forecast (No Priors)")
            contour_colors.append("gray")
        
        # Optional: parse and load Cobaya chains if directory is given
        if cobaya_chain_dir is not None:
            if sampled_params is None:
                sampled_params = param_names
                
            all_weights = []
            all_loglikes = []
            param_tracks = {p: [] for p in sampled_params}
            
            for i in range(num_chains):
                chain_path = os.path.join(cobaya_chain_dir, f"chain_task_{i}.txt")
                if os.path.exists(chain_path):
                    data = np.loadtxt(chain_path)
                    burn = int(burn_in_fraction * len(data))
                    
                    all_weights.append(data[burn:, 0])
                    all_loglikes.append(data[burn:, 1])
                    for idx, param in enumerate(sampled_params):
                        col_idx = idx + 2  # Skip weight and loglike columns
                        param_tracks[param].append(data[burn:, col_idx])
                else:
                    print(f"Warning: {chain_path} not found. Skipping.")
            
            if len(all_weights) == 0:
                raise FileNotFoundError(f"No valid chain files found in {cobaya_chain_dir}")
                
            combined_samples = np.column_stack([np.concatenate(param_tracks[p]) for p in sampled_params])
            cobaya_labels = [latex_labels.get(p, p) for p in sampled_params]
            
            mcmc_samples = MCSamples(
                samples=combined_samples,
                weights=np.concatenate(all_weights),
                loglikes=np.concatenate(all_loglikes),
                names=sampled_params,
                labels=cobaya_labels,
                settings={'ignore_rows': 0.0}
            )
            
            raw_datasets.append(mcmc_samples)
            legend_labels.append("Cobaya MCMC")
            contour_colors.append("darkblue")

        # Optional: Sample from uniform priors and plot as a faint background layer
        if plot_prior and uniform_priors is not None:
            
            # Generate uniform random samples covering the full prior range
            prior_samples_dict = {}
            for p in param_names:
                if p in uniform_priors:
                    p_min, p_max = uniform_priors[p]
                    # Match the exact number of samples for smooth shading
                    prior_samples_dict[p] = np.random.uniform(p_min, p_max, size=num_samples)
                else:
                    # Fallback if a parameter doesn't have an explicit prior bound defined
                    # (e.g., grab its current axes limits or use a wide placeholder)
                    fiducial = float(self.fiducial_dict[p])
                    prior_samples_dict[p] = np.random.uniform(fiducial * 0.5, fiducial * 1.5, size=num_samples)
                    
            prior_df = pd.DataFrame(prior_samples_dict)
            
            # Turn it into a GetDist MCSamples dataset
            prior_dataset = MCSamples(
                samples=prior_df.values, 
                names=param_names, 
                labels=labels, 
                name_tag="Uniform Priors"
            )
            
            # Insert it at the BEGINNING of your lists so it renders in the background
            raw_datasets.insert(0, prior_dataset)
            legend_labels.insert(0, "Uniform Priors")
            contour_colors.insert(0, "lightgray") # Faint color
        
        # 5. Create GetDist Subplot Plotter and generate the grid
        n_params = len(param_names)
        g = plots.get_subplot_plotter(width_inch=2.5 * n_params)
        
        # Extract fiducial values in correct order for markers
        fiducial_vals = {p: float(self.fiducial_dict[p]) for p in param_names if p in self.fiducial_dict}

        # Loop through each dataset and update the parameter labels manually
        for dataset in raw_datasets:
            for param_name, latex_string in latex_labels.items():
                # Get the list of names and check using 'in'
                if param_name in [p.name for p in dataset.paramNames.names]:
                    dataset.paramNames.parWithName(param_name).label = latex_string

        #### NEW -- CHECK
        #### IS THIS REALLY SLOW?
        # loop through datasets and convert from sampled_params to plot_params
        ### 
        print("Raw datasets: ", raw_datasets)
        plot_datasets = []
        raw_idx = {param: idx for idx, param in enumerate(sampled_params)}
        
        for dataset in raw_datasets:
            # If it's the analytical Fisher forecast object (GaussianND), draw mock samples
            if hasattr(dataset, 'sample'):
                samples = dataset.sample(100000) 
            else:
                # If it's a loaded MCMC chain (MCSamples), read the raw samples directly
                samples = dataset.samples
                        
            # Perform the column transformation if bases differ
            if list(sampled_params) != list(plot_params):
                projected_columns = []
                for p in plot_params:
                    
                    if p.lower() == 'omega_m':
                        b_col = raw_idx.get('omega_b') or raw_idx.get('Omega_b')
                        c_col = raw_idx.get('omega_c') or raw_idx.get('omega_cdm') or raw_idx.get('Omega_c')
                        projected_columns.append(samples[:, b_col] + samples[:, c_col])
                        
                    elif p.lower() == 'omega_lambda':
                        b_col = raw_idx.get('omega_b') or raw_idx.get('Omega_b')
                        c_col = raw_idx.get('omega_c') or raw_idx.get('omega_cdm') or raw_idx.get('Omega_c')
                        k_col = raw_idx.get('omega_k') or raw_idx.get('Omega_k')
                        
                        # Fallback to 0 if omega_k isn't part of the sampled parameters
                        k_samples = samples[:, k_col] if k_col is not None else 0.0
                        
                        projected_columns.append(1.0 - samples[:, b_col] - samples[:, c_col] - k_samples)
                    else:
                        projected_columns.append(samples[:, raw_idx[p]])
                        
                samples = np.column_stack(projected_columns)

            # Package up into GetDist format
            #### HOW DO I ADD A SUITABLE NAME TAG
            ### ARE THE LABELS ETC STILL CORRECT? -- PROBABLY NOT, RIGHT?
            mcsamples = MCSamples(
                samples=samples, 
                names=plot_params, 
            )
        
            plot_datasets.append(mcsamples)
        
        # Rebuild markers dynamically to match your new plot basis columns
        fiducial_vals = {}
        for p in plot_params:
            p_lower = p.lower()
            if p_lower == 'omega_m':
                fiducial_vals[p] = float(self.fiducial_dict['Omega_b']) + float(self.fiducial_dict['Omega_c'])
            elif p_lower == 'omega_lambda':
                b_val = float(self.fiducial_dict['Omega_b'])
                c_val = float(self.fiducial_dict['Omega_c'])
                k_val = float(self.fiducial_dict.get('Omega_k', 0.0))
                fiducial_vals[p] = 1.0 - b_val - c_val - k_val
            elif p in self.fiducial_dict:
                fiducial_vals[p] = float(self.fiducial_dict[p])
        
        g.triangle_plot(
            plot_datasets,
            params=plot_params,
            filled=True,
            contour_colors=contour_colors,
            legend_labels=legend_labels,
            markers=fiducial_vals,
            title_limit=None
        )
        
        if g.subplots is not None and g.subplots.size > 0:
            plt.subplots_adjust(top = 0.85)
            plt.figtext(0.15, 1, title, fontsize=16, ha='left', va='bottom')
            
        if save_plot:
            # Ensure the directory path exists safely
            if plot_folder and not os.path.exists(plot_folder):
                os.makedirs(plot_folder)
                print(f"Created directory: {plot_folder}")
            
            clean_filename = title.lower().replace(" ", "_").replace(".", "").replace(",", "") + ".pdf"
            full_save_path = os.path.join(plot_folder, clean_filename)
            
            # Check if GetDist plotter 'g' exists in local variables to use its native exporter
            if 'g' in locals():
                g.export(full_save_path)
            else:
                # Fallback to standard matplotlib if 'g' isn't explicitly defined
                plt.savefig(full_save_path, bbox_inches='tight', dpi=300)
                
            print(f"Plot successfully saved to: {full_save_path}")

        # Force clean math symbols for the table rows across all datasets
        latex_label_map = {
            'Omega_m': r'\Omega_m',
            'A_s': r'A_s',
            'w0': r'w_0',
            'wa': r'w_a',
            'h': r'h',
            'Omega_b': r'\Omega_{b}',
            'Omega_c': r'\Omega_{c}',
            'n_s': r'n_s',
            'Omega_k': r'\Omega_k',
            'Omega_lambda': r'\Omega_{\lambda}',
            'Neff': r'N_\mathrm{eff}',
            'm_nu': r'm_\mathrm{nu}',
            'T_CMB': r'T_\mathrm{CMB}'
        }

        ##### MODIFY TO PRINT PLOTTED PARAMS
        # Quantitative Parameter Comparison (Table Output)
        if print_summary:
            # Initialize the single master table header
            md_lines = [
                "| Parameter | Dataset / Model | 1-Sigma (68%) | 2-Sigma (95%) |",
                "| :--- | :--- | :---: | :---: |"
            ]
            
            for param in param_names:
                display_label = latex_label_map.get(param, param)
                param_idx = self.desired_params.index(param)
                fiducial_val = float(self.fiducial_dict[param])
                
                # Track the first row for this parameter block to display its name
                first_row_for_param = True

                # Helper function to dynamically convert any number into a clean LaTeX exponent string
                def format_value(val, sig):
                    # Check if either the value or the error falls outside [0.001, 99]
                    if abs(val) > 99 or abs(sig) > 99 or (0 < abs(val) < 0.001) or (0 < abs(sig) < 0.001):
                        # Convert to scientific notation (e.g., "2.10e-09" or "8.76e+09")
                        val_str = f"{val:.2e}"
                        sig_str = f"{sig:.2e}"
                        
                        # Split base and exponent: "2.10e-09" -> "2.10", "-09"
                        v_base, v_exp = val_str.split('e')
                        s_base, s_exp = sig_str.split('e')
                        
                        # Clean up sign/leading zeros in exponents (e.g., "-09" -> "-9", "+04" -> "4")
                        v_exp = int(v_exp)
                        s_exp = int(s_exp)
                        
                        # If they share the exact same exponent, group them cleanly like: (2.10 \pm 1.19) \cdot 10^{-9}
                        if v_exp == s_exp:
                            return f"({v_base} \\pm {s_base}) \\cdot 10^{{{v_exp}}}"
                        else:
                            # If exponents are different, print them individually
                            return f"{v_base} \\cdot 10^{{{v_exp}}} \\pm {s_base} \\cdot 10^{{{s_exp}}}"
                    else:
                        # Fall back to your standard readable decimal format
                        dec = 4 if sig < 0.01 else 3 
                        return f"{val:.{dec}f} \\pm {sig:.{dec}f}"

                for dataset, label in zip(raw_datasets, legend_labels):
                    
                    # --- CASE 1: Analytical Covariance Matrix for the Prior-less Forecast ---
                    if label == "Fisher Forecast (No Priors)":
                        sigma_1 = np.sqrt(self.cov[param_idx, param_idx])
                                
                        # Generate strings using our universal formatting rules
                        val_sig_1 = format_value(fiducial_val, sigma_1)
                        val_sig_2 = format_value(fiducial_val, 2.0 * sigma_1)
                        
                        # Build the final strings making sure the math label is ALWAYS attached
                        str_1sig = f"${display_label} = {val_sig_1}$"
                        str_2sig = f"${display_label} = {val_sig_2}$"
                    
                    # --- CASE 2: GetDist Objects (Fisher With Priors & Cobaya MCMC Chains) ---
                    else:
                        val_1sig = dataset.getInlineLatex(param, limit=1)
                        val_2sig = dataset.getInlineLatex(param, limit=2)
                        
                        # If GetDist included an '=', split it to throw away its broken label (e.g., 'Omegam')
                        if "=" in val_1sig:
                            val_1sig = val_1sig.split("=")[-1].strip()
                        if "=" in val_2sig:
                            val_2sig = val_2sig.split("=")[-1].strip()
                        
                        # Rebuild the string using your beautiful latex_label_map entry
                        str_1sig = f"${display_label} = {val_1sig}$"
                        str_2sig = f"${display_label} = {val_2sig}$"
                            
                    # FIX: Moved outside the except block so every model gets appended
                    param_col = f"**${display_label}$**" if first_row_for_param else ""
                    md_lines.append(f"| {param_col} | {label} | {str_1sig} | {str_2sig} |")
                    first_row_for_param = False
                            
                # Add a visual divider line between parameter blocks
                md_lines.append("| --- | --- | --- | --- |")
        
            # Render the unified master table cleanly in the notebook
            display(Markdown("\n".join(md_lines)))

            print("")
            print("")
            
        return g
