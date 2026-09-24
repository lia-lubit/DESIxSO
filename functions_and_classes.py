# import tools

# standard library
import os
import sys
import inspect
import yaml
import time
import types
import glob
import re
from copy import copy

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

# get edges for ell bins
def get_ell_bin_edges(l_min, n_ell, binsize, logarithmic=False):

    l_max = l_min + n_ell

    if logarithmic:
        num_bins = int(np.ceil(n_ell / binsize))
        edges = np.unique(np.geomspace(l_min, l_max, num_bins + 1).astype(int))
        edges[-1] = max(edges[-1], l_max)
    else:
        starts = np.arange(l_min, l_max, binsize)
        edges = np.append(starts, l_max)

    return edges

# map ell index to its bin
def ell_to_bin_index(ell, edges):
    return int(np.searchsorted(edges, ell, side='right') - 1) if ell < edges[-1] else len(edges) - 1
    
def load_txt_fisher_matrices(file_pattern="Fisher Forecasts/Fisher Matrices/*_Binsize=*.txt"):
    
    files = sorted(glob.glob(file_pattern))
    fisher_dict = {}
    param_names_dict = {}
    
    for filepath in files:
        match = re.search(r'Binsize=(\d+)', filepath)
        if match:
            binsize = int(match.group(1))

            print("Filepath: ", filepath)
            # Read header text directly to parse parameter names
            param_names = []
            with open(filepath, 'r') as f:
                lines = [f.readline(), f.readline()]
                for line in lines:
                    clean = line.strip('# \n')
                    if clean and not clean.startswith("Fiducial"):
                        param_names = clean.split()
            
            try:
                # Force numpy to skip the 2 text header rows
                matrix = np.loadtxt(filepath, skiprows=2)
                if matrix.ndim == 2 and matrix.size > 0:
                    fisher_dict[binsize] = matrix
                    param_names_dict[binsize] = param_names
            except Exception as e:
                print(f"Failed to load {filepath}: {e}")
                    
    return fisher_dict, param_names_dict

def analyze_fisher_matrices(fisher_dict):

    binsizes = sorted(fisher_dict.keys())
    
    results = {
        'binsize': [],
        'trace': [],
        'log_det': [],
        'marginal_errors': [],
        'unmarginal_errors': []
    }
    
    print(f"{'Bin Size':<10} | {'Trace F':<14} | {'log det(F)':<12} | Status")
    print("-" * 52)
    
    for b in binsizes:
        F = fisher_dict[b]
        
        tr = np.trace(F)
        sign, logdet = np.linalg.slogdet(F)
        
        # Unmarginalized error (1 / sqrt(F_ii))
        unmarg_errs = 1.0 / np.sqrt(np.diag(F))
        
        # Marginalized error sqrt((F^-1)_ii)
        try:
            Cov = np.linalg.inv(F)
            marg_errs = np.sqrt(np.diag(Cov))
            status = "OK"
        except np.linalg.LinAlgError:
            marg_errs = np.full(F.shape[0], np.nan)
            status = "Singular (Inv Failed)"
            
        results['binsize'].append(b)
        results['trace'].append(tr)
        results['log_det'].append(logdet)
        results['marginal_errors'].append(marg_errs)
        results['unmarginal_errors'].append(unmarg_errs)
        
        print(f"{b:<10} | {tr:<14.4e} | {logdet:<12.4f} | {status}")
        
    return results

def plot_fisher_diagnostics(results, param_names=None):

    binsizes = np.array(results['binsize'])
    traces = np.array(results['trace'])
    marginal_errs = np.array(results['marginal_errors'])
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. Mode Density Check: Trace vs Delta_ell
    axes[0].plot(binsizes, traces, 'o-', color='navy', label=r'Observed $\text{Tr}(F)$')
    
    axes[0].set_xlabel(r'Bin Size ($\Delta\ell$)')
    axes[0].set_ylabel(r'Fisher Trace $\text{Tr}(F)$')
    axes[0].set_title('Fisher Trace vs Bin Size (Mode Counting Diagnostic)')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # 2. Marginalized 1D Uncertainty vs Bin Size
    if marginal_errs.ndim > 1:
        num_params = marginal_errs.shape[1]
        if param_names is None or len(param_names) != num_params:
            param_names = [f'Param {i}' for i in range(num_params)]
            
        for i, name in enumerate(param_names):
            axes[1].plot(binsizes, marginal_errs[:, i], 's--', label=name)
            
    axes[1].set_xlabel(r'Bin Size ($\Delta\ell$)')
    axes[1].set_ylabel(r'Marginalized Uncertainty $\sigma(\theta)$')
    axes[1].set_title('1D Parameter Constraints vs Bin Size')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
def sample_fisher_with_uniform_priors(
    fiducial_cosmology,
    sampled_params,
    covariance,
    name_tag="Fisher",
    uniform_priors=None,
    num_samples=200000,
):
    """Generates GetDist MCSamples from a Fisher covariance matrix centered on fiducial cosmology."""
    # Helper to resolve fiducial value for each parameter
    def get_fid_val(p_name):
        val = None
        if isinstance(fiducial_cosmology, dict):
            val = fiducial_cosmology.get(p_name)
        elif hasattr(fiducial_cosmology, p_name):
            val = getattr(fiducial_cosmology, p_name)
        elif hasattr(fiducial_cosmology, "__getitem__"):
            try:
                val = fiducial_cosmology[p_name]
            except Exception:
                val = None

        if val is None:
            raise ValueError(f"Parameter '{p_name}' not found in fiducial_cosmology.")

        # Ensure scalar float (handles PyCCL array attributes like m_nu)
        if isinstance(val, (list, tuple, np.ndarray)):
            val = np.sum(val)
        return float(val)

    fiducial_values = np.array([get_fid_val(p) for p in sampled_params])

    # Ensure covariance is a numpy 2D square matrix
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"Covariance matrix must be square, got shape {covariance.shape}.")

    # Draw rapid multivariate normal samples based on the Fisher covariance
    samples = np.random.multivariate_normal(
        fiducial_values, covariance, size=num_samples
    )

    # Apply uniform priors (hard cuts) if specified
    if uniform_priors is not None:
        mask = np.ones(num_samples, dtype=bool)
        for i, p in enumerate(sampled_params):
            if p in uniform_priors:
                p_min, p_max = uniform_priors[p]
                mask &= (samples[:, i] >= p_min) & (samples[:, i] <= p_max)
        samples = samples[mask]

    # Convert into a GetDist MCSamples object for seamless plotting
    param_labels = [p for p in sampled_params]

    mcsamples = MCSamples(
        samples=samples,
        names=sampled_params,
        labels=param_labels,
        name_tag=name_tag,
    )

    return mcsamples

def plot_Fisher_from_matrices(
    fiducial_cosmology,
    Fisher_chains=None,
    DESI_chains=None,
    title=None,
    sampled_params=None,
    params_to_plot=None,
    save_plot=False,
    plot_folder="plots/Fisher Forecasts",
    num_samples=200000,
    filled=[True, True, True, True, True, True],
    line_styles=['-', '-', '-', '-', '-', '-']
):
    """Plots Fisher and DESI chains on the same triangle plot with automatic parameter conversion."""
    # Standardize parameters to plot
    plot_params = (
        params_to_plot if params_to_plot is not None else sampled_params
    )
    if plot_params is None:
        plot_params = ["Omega_m", "w0", "wa", "h", "n_s", "A_s"]

    raw_datasets = []
    legend_labels = []
    contour_colors = []

    latex_labels = {
        "Omega_m": r"\Omega_\mathrm{m}",
        "Omega_b": r"\Omega_\mathrm{b}",
        "Omega_k": r"\Omega_\mathrm{k}",
        "Omega_c": r"\Omega_\mathrm{c}",
        "Omega_lambda": r"\Omega_\mathrm{\Lambda}",
        "wa": r"w_a",
        "w0": r"w_0",
        "h": r"h",
        "logA_s": r"logA_\mathrm{s}",
        "A_s": r"A_\mathrm{s}",
        "n_s": r"n_\mathrm{s}",
        "Neff": r"N_\mathrm{eff}",
        "m_nu": r"m_\mathrm{nu}",
        "T_CMB": r"T_\mathrm{CMB}",
    }
    labels = [latex_labels.get(p, p) for p in plot_params]

    # Color palettes for Fisher vs DESI chains
    fisher_colors = ["purple"]
    desi_colors = ["blue", "darkorange", "green", "pink"]


    # 2. Process DESI chains
    if DESI_chains is not None:
        chains_list = (
            DESI_chains if isinstance(DESI_chains, list) else [DESI_chains]
        )

        # If there are multiple DESI chains, put the first chain at the end so it plots on top
        if len(chains_list) > 1:
            reordered_desi = [chains_list[1], chains_list[2], chains_list[3], chains_list[0]]
        else:
            reordered_desi = chains_list
            
        for idx, dc in enumerate(chains_list):
            raw_datasets.append(dc)
            label = getattr(
                dc, "label", "DESI DR2" if idx == 0 else f"DESI DR2 #{idx+1}"
            )
            legend_labels.append(label)
            contour_colors.append(desi_colors[idx % len(desi_colors)])

    # 1. Process Fisher chains
    if Fisher_chains is not None:
        chains_list = (
            Fisher_chains
            if isinstance(Fisher_chains, list)
            else [Fisher_chains]
        )
        for idx, fc in enumerate(chains_list):
            raw_datasets.append(fc)
            label = getattr(
                fc,
                "label",
                "Fisher Forecast" if idx == 0 else f"Fisher #{idx+1}",
            )
            legend_labels.append(label)
            contour_colors.append(fisher_colors[idx % len(fisher_colors)])
            
    if not raw_datasets:
        raise ValueError(
            "No valid datasets provided in Fisher_chains or DESI_chains."
        )

    # Helper function to resolve parameter value/column from cosmology object or dict
    def get_fiducial_val(p_name, fallback=0.0):
        val = None
        if isinstance(fiducial_cosmology, dict):
            val = fiducial_cosmology.get(p_name, fallback)
        elif hasattr(fiducial_cosmology, p_name):
            val = getattr(fiducial_cosmology, p_name)
        elif hasattr(fiducial_cosmology, "__getitem__"):
            try:
                val = fiducial_cosmology[p_name]
            except Exception:
                val = fallback
        else:
            val = fallback

        if val is None:
            val = fallback

        # Convert array/list values (e.g. m_nu = [0.06, ...]) to a scalar float
        if isinstance(val, (list, tuple, np.ndarray)):
            val = np.sum(val)

        return float(val)

    # Helper function to extract, compute, and re-order parameter columns per dataset
    def extract_param_samples(dataset, target_params):
        if hasattr(dataset, "paramNames") and dataset.paramNames is not None:
            existing_names = [p.name for p in dataset.paramNames.names]
        elif hasattr(dataset, "names"):
            existing_names = list(dataset.names)
        else:
            existing_names = (
                sampled_params if sampled_params is not None else target_params
            )

        samples = dataset.samples if hasattr(dataset, "samples") else dataset
        chain_len = samples.shape[0]
        idx_map = {name: i for i, name in enumerate(existing_names)}

        def get_col(p_name):
            if p_name in idx_map:
                return samples[:, idx_map[p_name]]
            fid_val = get_fiducial_val(p_name, fallback=0.0)
            return np.full(chain_len, fid_val)

        projected_cols = []
        for p in target_params:
            if p in idx_map:
                projected_cols.append(samples[:, idx_map[p]])
            elif p == "Omega_m":
                b = get_col("Omega_b")
                c = get_col("Omega_c")
                h = get_col("h")
                m_nu = get_col("m_nu")
                om_nu = m_nu / (h * h * 93.15) if np.any(m_nu != 0) else 0.0
                projected_cols.append(b + c + om_nu)
            elif p == "Omega_lambda":
                b = get_col("Omega_b")
                c = get_col("Omega_c")
                k = get_col("Omega_k")
                h = get_col("h")
                m_nu = get_col("m_nu")
                om_nu = m_nu / (h * h * 93.15) if np.any(m_nu != 0) else 0.0
                projected_cols.append(1.0 - b - c - k - om_nu)
            else:
                projected_cols.append(get_col(p))

        return np.column_stack(projected_cols)

    # 3. Align datasets into GetDist MCSamples format
    plot_datasets = []
    for idx, ds in enumerate(raw_datasets):
        aligned_samples = extract_param_samples(ds, plot_params)
        weights = (
            ds.weights
            if hasattr(ds, "weights") and ds.weights is not None
            else None
        )

        mcsamples = MCSamples(
            samples=aligned_samples,
            weights=weights,
            names=plot_params,
            labels=labels,
            label=legend_labels[idx],
        )
        plot_datasets.append(mcsamples)

    # 4. Generate GetDist Triangle Plot
    n_params = len(plot_params)
    g = plots.get_subplot_plotter(width_inch=2.2 * n_params)
    g.settings.legend_fontsize = 12

    # Extract fiducial values for plot markers
    fiducial_vals = {}
    for p in plot_params:
        val = get_fiducial_val(p, fallback=None)
        if val is not None:
            fiducial_vals[p] = float(val)

    if contour_colors:
        line_args = [{'color': c} for c in contour_colors]
    else:
        line_args = None
        
    g.triangle_plot(
        plot_datasets,
        params=plot_params,
        filled=filled,                       # Mix of filled and contour outlines
        line_args=line_args,
        contour_colors=contour_colors[: len(plot_datasets)],
        legend_ncol=1,
        subplots_hide_invisible=True,
        kwargs={
            # Adjust x (0.55-0.60) to push it to the top right
            'bbox_to_anchor': (0.60, 0.65, 0.35, 0.25),
            'loc': 'upper right',  # Anchors the legend box's top-right corner
            'mode': 'expand',     # Forces fixed (width, height) box dimensions
            'borderaxespad': 0.0, # Removes extra automatic padding shifts
        },
        markers=fiducial_vals if len(fiducial_vals) > 0 else None,
        title_limit=None,
    )
    
    if g.subplots is not None and g.subplots.size > 0:
        plt.subplots_adjust(top=0.90)

    if save_plot:
        os.makedirs(plot_folder, exist_ok=True)
        filename = f"{title.replace(' ', '_')}.png"
        plt.savefig(
            os.path.join(plot_folder, filename),
            bbox_inches="tight",
            dpi=1200,
        )

    return g
    
# get derivatives for select parameters 
def get_partial_derivative(fiducial_values, param1, param2):
    
    if param1 == param2:
        return 1
    
    elif param1 == 'omega_c':
        if param2 == 'Omega_c':
            return (fiducial_values['h'] ** 2)
        elif param2 == 'h':
            return (2 * fiducial_values['h'] * fiducial_values['Omega_c'])
            
    elif param1 == 'omega_b':
        if param2 == 'Omega_b':
            return (fiducial_values['h'] ** 2)
        elif param2 == 'h':
            return (2 * fiducial_values['h'] * fiducial_values['Omega_b'])
                    
    elif param1 == 'H':
        if param2 == 'h':
            return 100
                    
    else:
        return 0
        
# project matrix from one set of parameters to another
# e.g. omega_b H to Omega_b h
#### CHECK
def project_fisher_matrix(F_raw, fiducial_values, sampled_params, new_params):

    print("WARNING: Our get_partial_derivative function is only equipped to handle Omega_m, Omega_b, omega_b, Omega_c, omega_c, Omega_k, H, h, A_s, n_s, Neff, m_nu, and T_CMB. If you have passed over parameters to it DO NOT TRUST THE RESULTS.")
    
    # if the bases are identical, no projection is needed
    if list(sampled_params) == list(plot_params):
        return F_raw

    # build Jacobian matrix 
    n_sampled = len(sampled_params)
    n_plotted = len(new_params)
    J = np.zeros((n_sampled, n_plotted))
    
    for i in range(n_sampled):
        for j in range(n_plotted):
            # each entry is the partial derivative of the old parameter wrt the new parameter
            J[i][j] = self.get_partial_derivative(fiducial_values, sampled_params[i], new_params[j])
            
    # Transform: F_new = J^T @ F_raw @ J
    F_projected = J.T @ F_raw @ J

    return F_projected

def find_key_recursive(data, target_key):
    """Recursively searches for a key in a nested dictionary/list structure."""
    if isinstance(data, dict):
        if target_key in data:
            return data[target_key]
        for key, value in data.items():
            result = find_key_recursive(value, target_key)
            if result is not None:
                return result
    elif isinstance(data, list):
        for item in data:
            result = find_key_recursive(item, target_key)
            if result is not None:
                return result
    return None

# plot traces and contour plots from Cobaya, including reference points
# mark if the run was incomplete, but assume it was complete unless otherwise stated
def plot_cobaya_mcmc_results(chain_dir, yaml_path, sampled_params, plot_params, num_chains=4, burn_in_fraction=0.2, output_dir='plots', complete=True, plot_triangle = True, plot_traces = True, print_summary = False, save_triangle = False, save_traces = False):

    # fall back to plotting sampled parameteres
    if plot_params is None:
        plot_params = sampled_params
        
    print(f"Reading fiducial values from: {yaml_path}")
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    fiducial_block = find_key_recursive(config, 'fiducial_cosmology_params')
    if fiducial_block is None:
        raise KeyError("Could not find 'fiducial_cosmology_params' anywhere inside the provided YAML file.")

    fiducial_vals = {}
    for p in fiducial_block:
        fiducial_vals[p] = float(fiducial_block[p])
        
    # Process Chains, Extract Initial Points, and Apply Burn-In
    all_weights = []
    all_loglikes = []
    param_tracks = {p: [] for p in sampled_params}
    initial_points = []
    raw_chain_data = []

    print(f"Processing {num_chains} chains from: {chain_dir}")
    chain_lengths=[]
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
            chain_lengths.append(len(data[burn:]))
        else:
            print(f"Warning: {chain_path} not found. Skipping.")

    combined_samples = np.column_stack([np.concatenate(param_tracks[p]) for p in sampled_params])
    
    latex_labels = {
        'Omega_m': r'\Omega_\mathrm{m}',
        'Omega_b': r'\Omega_\mathrm{b}',
        'Omega_c': r'\Omega_\mathrm{c}',
        'Omega_k': r'\Omega_\mathrm{k}',
        'Omega_lambda': r'\Omega_\mathrm{\Lambda}',
        'wa': r'w_a',
        'w0': r'w_0',
        'h': r'h',
        'A_s': r'A_\mathrm{s}',
        'logA_s': r'logA_\mathrm{s}',
        'n_s': r'n_\mathrm{s}',
        'Neff': r'N_\mathrm{eff}',
        'm_nu': r'm_\mathrm{nu}',
        'T_CMB': r'T_\mathrm{CMB}'
    }

    labels = [latex_labels.get(p, p) for p in plot_params]

    # add extra parameters from plot_params to the chain dict, etc.
    chain_columns = list(sampled_params)

    if list(sampled_params) != list(plot_params):
        chain_length = len(combined_samples)
        indices = {p: i for p, i in zip(sampled_params, range(len(sampled_params)))}

        h_data = combined_samples[:, indices['h']] if indices.get('h') is not None else np.full(chain_length, fiducial_vals.get('h'))
        omega_b_data = combined_samples[:, indices['Omega_b']] if indices.get('Omega_b') is not None else np.full(chain_length, fiducial_vals.get('Omega_b'))
        omega_c_data = combined_samples[:, indices['Omega_c']] if indices.get('Omega_c') is not None else np.full(chain_length, fiducial_vals.get('Omega_c'))
        m_nu_data = combined_samples[:, indices['m_nu']] if indices.get('m_nu') is not None else np.full(chain_length, fiducial_vals.get('m_nu'))
        omega_k_data = combined_samples[:, indices['Omega_k']] if indices.get('Omega_k') is not None else np.full(chain_length, fiducial_vals.get('Omega_k', 0.0))

        # Compute derived arrays
        omega_nu_data = m_nu_data / ((h_data ** 2) * 93.15)
        omega_m_data = omega_b_data + omega_c_data + omega_nu_data
        omega_lambda_data = 1.0 - omega_m_data - omega_k_data

        # Stack columns and calculate companion dictionary entries and chains
        for name, data_array in [('Omega_m', omega_m_data), ('Omega_lambda', omega_lambda_data)]:
            if name in plot_params and name not in chain_columns:
                combined_samples = np.column_stack([combined_samples, data_array])
                chain_columns.append(name)
                labels.append(latex_labels.get(name, name))
                
                # --- CASE 1: OMEGA_M ---
                if name == 'Omega_m':
                    # fiducial values dictionary
                    h_f = float(fiducial_block.get('h'))
                    m_nu_f = float(fiducial_block.get('m_nu'))
                    fiducial_vals['Omega_m'] = float(fiducial_block.get('Omega_c')) + float(fiducial_block.get('Omega_b')) + (m_nu_f / ((h_f ** 2) * 93.15))
                    
                    # initial points dictionary
                    for pt in initial_points:
                        omega_c = pt.get('Omega_c', fiducial_vals.get('Omega_c'))
                        omega_b = pt.get('Omega_b', fiducial_vals.get('Omega_b'))
                        m_nu_pt = pt.get('m_nu', fiducial_vals.get('m_nu'))
                        h_pt    = pt.get('h', fiducial_vals.get('h'))
                        
                        pt['Omega_m'] = omega_c + omega_b + (m_nu_pt / ((h_pt ** 2) * 93.15))

                    #chains
                    if name not in param_tracks:
                        param_tracks[name] = []
                    start_idx = 0
                    for i in range(num_chains):
                        end_idx = start_idx + chain_lengths[i]
                        param_tracks[name].append(omega_m_data[start_idx:end_idx])
                        start_idx = end_idx

                # OMEGA_LAMBDA
                elif name == 'Omega_lambda':
                    # fiducial values dictionary
                    h_f = float(fiducial_block.get('h'))
                    m_nu_f = float(fiducial_block.get('m_nu'))
                    om_f = float(fiducial_block.get('Omega_c')) + float(fiducial_block.get('Omega_b')) + (m_nu_f / ((h_f ** 2) * 93.15))
                    ok_f = float(fiducial_block.get('Omega_k', 0.0))
                    fiducial_vals['Omega_lambda'] = 1.0 - om_f - ok_f

                    # initial points dictionary
                    for pt in initial_points:
                        omega_c = pt.get('Omega_c', fiducial_vals.get('Omega_c'))
                        omega_b = pt.get('Omega_b', fiducial_vals.get('Omega_b'))
                        m_nu_pt = pt.get('m_nu', fiducial_vals.get('m_nu'))
                        h_pt    = pt.get('h', fiducial_vals.get('h'))
                        ok_pt   = pt.get('Omega_k', fiducial_vals.get('Omega_k', 0.0))
                        
                        # Explicitly compute total matter for this point from scratch
                        om_pt = omega_c + omega_b + (m_nu_pt / ((h_pt ** 2) * 93.15))
                        pt['Omega_lambda'] = 1.0 - om_pt - ok_pt

                    # chains
                    if name not in param_tracks:
                        param_tracks[name] = []
                    start_idx = 0
                    for i in range(num_chains):
                        end_idx = start_idx + chain_lengths[i]
                        param_tracks[name].append(omega_lambda_data[start_idx:end_idx])
                        start_idx = end_idx
                        
    samples = MCSamples(
        samples=combined_samples,
        weights=np.concatenate(all_weights),
        loglikes=np.concatenate(all_loglikes),
        names=chain_columns,
        labels=labels,
        settings={'ignore_rows': 0.0}
    )

    peaks = {}
    for param in plot_params:
        density1D = samples.get1DDensity(param)
        peaks[param] = density1D.x[np.argmax(density1D.P)]

    # NAME STRIPPING & COMPLETION MODIFICATIONS
    os.makedirs(output_dir, exist_ok=True)
    raw_run_name = os.path.basename(os.path.normpath(chain_dir))
    clean_run_name = raw_run_name.replace('_likelihood', '').replace('likelihood', '')
    display_title = clean_run_name.replace('_', ' ')

    # Append tags dynamically based on the 'complete' flag state
    if not complete:
        file_suffix = "_(incomplete)"
        title_suffix = " (incomplete)"
    else:
        file_suffix = ""
        title_suffix = ""

    # TRIANGLE CONTOUR PLOT
    if plot_triangle:
        for param_name, latex_string in latex_labels.items():
            if samples.paramNames.hasParam(param_name):
                samples.paramNames.parWithName(param_name).label = latex_string
            
        g1 = plots.get_subplot_plotter(width_inch=2.5 * len(plot_params))
        g1.triangle_plot(
            [samples], 
            params=plot_params, 
            filled=True, 
            contour_colors=['darkblue'],
            title_limit=1,
            markers=fiducial_vals
        )
    
        for row in range(len(plot_params)):
            for col in range(row + 1):
                ax = g1.subplots[row, col]
                if ax is None:
                    continue
                p_row = plot_params[row]
                p_col = plot_params[col]
                if row == col:
                    ax.axvline(x=peaks[p_row], color='crimson', linestyle=':', alpha=0.8, label='MCMC Peak')
                    for idx, pt in enumerate(initial_points):
                        lbl = 'Initial Points' if (row == 0 and idx == 0) else ""
                        ax.axvline(x=pt[p_row], color='darkorange', linestyle='-', alpha=0.4, linewidth=1, label=lbl)
                    if row == 0:  
                        ax.legend(loc='upper right', fontsize=8)
                else:
                    for idx, pt in enumerate(initial_points):
                        lbl = 'Initial Points' if (row == len(plot_params)-1 and col == len(plot_params)-2 and idx == 0) else ""
                        ax.scatter(pt[p_col], pt[p_row], color='darkorange', marker='x', s=40, zorder=5, alpha=0.8, label=lbl)
                    if row == len(plot_params)-1 and col == len(plot_params)-2:
                        ax.legend(loc='upper right', fontsize=8)
    
        plt.suptitle(f"Marginalized Constraints: {display_title}{title_suffix}", y=1.02, fontsize=10)
    
        if save_triangle:
            triangle_save_path = os.path.join(output_dir, f"{clean_run_name}_triangle_plot{file_suffix}.pdf")
            g1.export(triangle_save_path)
            print(f"Saved triangle plot to: {triangle_save_path}")
        
        plt.show()
    
    # TRACE PLOTS
    if plot_traces:
        fig, axes = plt.subplots(len(plot_params), 1, figsize=(12, 3 * len(plot_params)), sharex=True)
        if len(plot_params) == 1:
            axes = [axes]

        for idx, param in enumerate(plot_params):
            ax = axes[idx]
            
            # Use .get() to safely check if the parameter key exists in the dictionary
            chains_list = param_tracks.get(param, None)
            
            if chains_list is not None:
                # Loop through each individual chain array stored under this key
                for i, chain_data in enumerate(chains_list):
                    ax.plot(chain_data, alpha=0.5, linewidth=0.8, label=f'Chain {i+1}')
            else:
                print(f"Warning: No trace data found for {param} in param_tracks. Skipping line plot.")
            
            # Add Peak and Fiducial lines
            if param in peaks:
                ax.axhline(y=peaks[param], color='crimson', linestyle=':', alpha=0.8, label=f'Peak: {peaks[param]:.4f}')
            
            if param in fiducial_vals:
                ax.axhline(y=fiducial_vals[param], color='black', linestyle='--', alpha=0.6, label=f'Fiducial: {fiducial_vals[param]:.4f}')
                
            param_label = latex_labels.get(param, param)
            ax.set_ylabel(f"${param_label}$")
            ax.legend(loc='upper right', fontsize=8, ncol=2) 
            ax.grid(True, alpha=0.3)
    
        axes[0].set_title(f"MCMC Trace Plots: {display_title}{title_suffix}", fontsize=12)
        axes[-1].set_xlabel('Step Number (After Burn-in)')
        
        plt.tight_layout()
    
        if save_traces:
            trace_save_path = os.path.join(output_dir, f"{clean_run_name}_traces{file_suffix}.png")
            plt.savefig(trace_save_path, dpi=300, bbox_inches='tight')
            print(f"Saved trace plot to: {trace_save_path}")
    
        plt.show()
        
    # PRINT SUMMARY 
    if print_summary:
        # Initialize the single master table header
        md_lines = [
            "| Parameter | 1-Sigma (68%) | 2-Sigma (95%) |",
            "| :--- | :---: | :---: |"
        ]
        
        for param in plot_params:
            display_label = latex_labels.get(param, param)
            fiducial_val = float(fiducial_vals[param])
            
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
    
    
            val_1sig = samples.getInlineLatex(param, limit=1)
            val_2sig = samples.getInlineLatex(param, limit=2)
            
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
            md_lines.append(f"| {param_col} | {str_1sig} | {str_2sig} |")
            first_row_for_param = False
                        
            # Add a visual divider line between parameter blocks
            md_lines.append("| --- | --- | --- |")
    
        # Render the unified master table cleanly in the notebook
        display(Markdown("\n".join(md_lines)))
    
        print("")
        print("")
        
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
    ell_values = np.logspace(np.log10(l_min), np.log10(n_ell + l_min), int(n_ell / 10))
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
        #cl = cosmology.angular_cl(cmb_lensing_tracer, cmb_lensing_tracer, ell_values, p_of_k_a = Pk2D_object)
        #cl_spectra[f'CC'] = cl
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
        l_max = n_ell + l_min

        # fix the error wherein it thinks m_nu is a list
        if hasattr(m_nu, '__len__') or isinstance(m_nu, (list, np.ndarray)):
            m_nu = np.sum(m_nu)

        # tau isn't currently in your cosmology object — grab it if present,
        # otherwise fall back to a default. Worth adding to `cosmology`
        # properly if you want it to vary.
        tau = cosmology.get('tau', 0.0544) if hasattr(cosmology, 'get') else 0.0544

        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=100.0 * h,
            ombh2=ombh2,
            omch2=omch2,
            omk=Omega_k,
            mnu=m_nu,
            nnu=Neff,
            tau=tau,
            TCMB=T_CMB,
        )
        pars.InitPower.set_params(As=A_s, ns=n_s)
        pars.set_dark_energy(w=w0, wa=wa, dark_energy_model='ppf')

        # match nonlinear treatment to whatever pyccl is using elsewhere,
        # if your primary spectra assume nonlinear lensing
        pars.NonLinear = camb.model.NonLinear_both

        pars.set_for_lmax(l_max, lens_potential_accuracy=1)

        results = camb.get_results(pars)

        lens_cls = results.get_lens_potential_cls(lmax=l_max, raw_cl=True)
        ls = np.arange(lens_cls.shape[0])
        clpp = lens_cls[:, 0]

        factor = (ls * (ls + 1) / 2.0)**2
        clkk_full = factor * clpp

        # interpolate onto the same ell_values grid used by your pyccl spectra,
        # so cl_spectra['CC'] is consistent with everything else in the dict
        clkk = np.interp(ell_values, ls, clkk_full)

        cl_spectra['CC'] = clkk


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
        l_max = n_ell + l_min

        # fix the error wherein it thinks m_nu is a list
        if hasattr(m_nu, '__len__') or isinstance(m_nu, (list, np.ndarray)):
                m_nu = np.sum(m_nu)        
        #Neff = float(Neff[0]) if isinstance(Neff, (list, np.ndarray)) else float(Neff)
        #T_CMB = float(T_CMB[0]) if isinstance(T_CMB, (list, np.ndarray)) else float(T_CMB)

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
##### ADD LOGARITHMIC BINNING
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

##### ADD LOGARITHMIC BINNING
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
def plot_spectra_from_dict(spectra_dict, title_prefix='Angular Power Spectrum', desired_spectra=None, plot_scaled=False, plot_linear=True, num_lens_bins = 0, num_source_bins = 0, l_min = 2):
    if not spectra_dict:
        print("No spectra to plot.")
        return

    # Get the ells from the first entry in spectra_dict
    # Assuming all spectra have the same ell values
    first_key = next(iter(spectra_dict))
    ells = np.arange(l_min, len(spectra_dict[first_key]) + l_min)

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
def build_data_vector(forecast_map, spectra_dict, n_ell, binsize, logarithmic=False, l_min=2):
    ells = np.arange(l_min, l_min + n_ell)
    observed_data_vector = []

    edges = get_ell_bin_edges(l_min, n_ell, binsize, logarithmic)
    bins = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    for pair in forecast_map.pairs:
        if pair in spectra_dict:
            unbinned_cls = spectra_dict[pair]
        elif pair[::-1] in spectra_dict:
            unbinned_cls = spectra_dict[pair[::-1]]
        else:
            print(f"spectrum for {pair} not found, assuming zero.")
            unbinned_cls = np.zeros(n_ell)

        binned_cls_for_pair = []
        for start_ell, end_ell in bins:
            idx_start = start_ell - l_min
            idx_end = end_ell - l_min

            bin_cls = unbinned_cls[idx_start:idx_end]
            bin_ells = ells[idx_start:idx_end]

            if len(bin_cls) == 0:
                continue

            weights = 2 * bin_ells + 1
            weighted_mean = np.sum(weights * bin_cls) / np.sum(weights)
            binned_cls_for_pair.append(weighted_mean)

        observed_data_vector.extend(binned_cls_for_pair)

    return np.array(observed_data_vector)
    
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

    return lens_tracers, galaxy_lensing_tracers, cmb_lensing_tracer

# build tracer dictionary
def build_tracer_dict(lens_tracers, galaxy_lensing_tracers, cmb_lensing_tracer):
    tracer_dict = {'kappa_c': cmb_lensing_tracer}
    
    for i, tr in enumerate(lens_tracers):
        tracer_dict[f'g{i+1}'] = tr
    for i, tr in enumerate(galaxy_lensing_tracers):
        tracer_dict[f'kappa_g{i+1}'] = tr

    return tracer_dict

# build noise dictionary
# shot noise needs to have as many entries as lens bins
# shape noise needs to have as many entries as source bins
def build_noise_dict(f_map, ells, shot_noise_lens=None, shape_noise_source=None, cmb_noise_kk=None, cmb_noise_TT=None, cmb_noise_EE=None):
    
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

    return noise_dict

# build spectra dictionary w/ or w/o emulator
#### CHECK
#### does this function actually consider f_map at all?
def build_spectra_dict(cosmo, f_map, tracer_dict, ells, noise_dict=None,
                        linear_emulator=None, boost_emulator=None,
                        cmb_primaries=False, pk_override=None):
    spectra_dict = {}

    if linear_emulator is not None and pk_override is None:
        a_grid = np.linspace(1/(1+5), 1.0, 20)
        z_grid = (1.0 / a_grid) - 1.0
        Pk2D_object = make_Pk2D(cosmo, linear_emulator=linear_emulator, boost_emulator=boost_emulator,
                                 z_arr=z_grid, cmin=3.13, eta_0=0.60)

    tracer_labels = list(tracer_dict.keys())
    for i, label1 in enumerate(tracer_labels):
        for j, label2 in enumerate(tracer_labels):
            key_fwd = (label1, label2)
            key_bwd = (label2, label1)
            if key_fwd in spectra_dict or key_bwd in spectra_dict:
                continue

            tracer1 = tracer_dict[label1]
            tracer2 = tracer_dict[label2]

            if pk_override is not None:
                # frozen P(k,a): cosmo still supplies distances/kernels via
                # tracer1/tracer2, but the power spectrum itself never moves
                Cl = ccl.angular_cl(cosmo, tracer1, tracer2, ells, p_of_k_a=pk_override)
            elif linear_emulator is not None:
                Cl = ccl.angular_cl(cosmo, tracer1, tracer2, ells, p_of_k_a=Pk2D_object)
            else:
                Cl = ccl.angular_cl(cosmo, tracer1, tracer2, ells)

            if label1 < label2:
                spectra_dict[(label1, label2)] = Cl
            else:
                spectra_dict[(label2, label1)] = Cl
            
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
        if hasattr(m_nu, '__len__') or isinstance(m_nu, (list, np.ndarray)):
            m_nu = np.sum(m_nu)
        #Neff = float(Neff[0]) if isinstance(Neff, (list, np.ndarray)) else float(Neff)
        #T_CMB = float(T_CMB[0]) if isinstance(T_CMB, (list, np.ndarray)) else float(T_CMB)

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

        # E-mode cross-correlations cannot be simply calculated with PyCCL or CAMB
        # they are negligable and are set to zero
        if 'kappa_c' in tracer_dict:
            spectra_dict[('E', 'kappa_c')] = np.zeros_like(ells) 
            spectra_dict[('T', 'kappa_c')] = np.zeros_like(ells) 
        
        for g_bin in lens_bins:
            spectra_dict[('E', g_bin)] = np.zeros_like(ells)     
            spectra_dict[('T', g_bin)] = np.zeros_like(ells)     

        for kg_bin in source_bins:
            spectra_dict[('E', kg_bin)] = np.zeros_like(ells)    
            spectra_dict[('T', kg_bin)] = np.zeros_like(ells)    

    # add noise terms
    if noise_dict is not None:
        for key_noise, noise_val in noise_dict.items():
            # generally only add noise to auto-spectra.
            if key_noise[0] == key_noise[1]:
                spectra_dict[key_noise] += noise_val
    
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

    if 'GT' in desired_spectra:  # Lens Galaxy Clustering x CMB Temperature
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
    f_sky_c, # sky fraction for cmb 
    f_sky_g, # sky fraction for lens galaxies
    f_sky_l, # sky fraction for source galaxies
    f_sky_c_g, # sky overlap for cmb and lens galaxies
    f_sky_c_l, # sky overlap for cmb and source galaxies
    f_sky_g_l, # sky overlap for source and lens galaxies
    f_sky_c_g_l, # sky overlap for cmb and source and lens galaxies
    l_min=2,
    n_ell=3000, 
    binsize=1,  
    logarithmic=False,
    shot_noise_lens=None,
    shape_noise_source=None,
    cmb_noise_kk=None,
    cmb_noise_TT=None,
    cmb_noise_EE=None,
    magnification_bias_lenses=None, 
    desired_spectra=None,
    linear_emulator=None,
    boost_emulator=None,
    cmb_primaries=False,
    z_max=6,
    n_chi=1024
):

    full_f_map = ForecastMap(n_lens=lens_data.shape[1]-1, n_src=source_data.shape[1]-1, l_min=l_min, n_ell=n_ell, desired_pairs = None, cmb_primaries = cmb_primaries)
    
    # Use the full range of unbinned ells for CCL calculations
    ells = np.arange(l_min, n_ell + l_min)
    
    cosmo.compute_growth()
    
    # build spectra
    lens_tracers, source_tracers, cmb_lensing_tracer = build_tracers_from_data(cosmo, lens_data, source_data, magnification_bias_lenses, z_max = z_max, n_chi = n_chi)
    tracer_dict = build_tracer_dict(lens_tracers, source_tracers, cmb_lensing_tracer)
    noise_dict = build_noise_dict(full_f_map, ells, shot_noise_lens, shape_noise_source, cmb_noise_kk, cmb_noise_TT = cmb_noise_TT, cmb_noise_EE = cmb_noise_EE)
    full_spectra_dict = build_spectra_dict(cosmo, full_f_map, tracer_dict, ells, noise_dict, linear_emulator=linear_emulator, boost_emulator=boost_emulator, cmb_primaries = cmb_primaries)

    # build covariance -- now pass the binsize to CovarianceMatrix
    full_cov = CovarianceMatrix(full_f_map, full_spectra_dict, f_sky_c, f_sky_g, f_sky_l, f_sky_c_g, f_sky_c_l, f_sky_g_l, f_sky_c_g_l, binsize=binsize, logarithmic=logarithmic)

    if desired_spectra is None:
        return full_cov, full_spectra_dict, full_f_map
    else:
        sliced_pairs = create_simplified_desired_pairs(lens_data.shape[1] - 1, source_data.shape[1] - 1, desired_spectra)
        sliced_f_map = ForecastMap(n_lens=lens_data.shape[1]-1, n_src=source_data.shape[1]-1, l_min=l_min, n_ell=n_ell, desired_pairs=sliced_pairs, cmb_primaries=cmb_primaries)
        # Loop over full_spectra_dict to preserve its original, chronological block order
        sliced_spectra_dict = {pair: full_spectra_dict[pair] for pair in full_spectra_dict if pair in sliced_pairs}
        sliced_cov = slice_matrix(full_cov, full_spectra_dict, full_f_map, binsize=binsize, logarithmic=logarithmic, desired_spectra=desired_spectra)
        return sliced_cov, sliced_spectra_dict, sliced_f_map

# slice vector and matrix given desired pairs
# note that the returned object is not a real CovarianceMatrix object, but it contains the necessary information
###### CHECK
def slice_matrix(
    cov_obj, 
    spectra_dict, 
    f_map, 
    binsize=1, 
    logarithmic=False,
    desired_spectra=None
):

    n_ell = f_map.n_ell
    l_min = f_map.l_min
    
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
                
                if p[0] > p[1]:
                    canonical_pair = (p[1], p[0])
                else:
                    canonical_pair = p
                    
                if canonical_pair not in processed_desired_pairs:
                    processed_desired_pairs.append(canonical_pair)
                    
            pairs_to_slice = processed_desired_pairs

    edges = get_ell_bin_edges(l_min, n_ell, binsize, logarithmic)
    n_bins = len(edges) - 1

    all_ranges = []
    final_sliced_pairs = []

    for pair in f_map.pairs:
        if pair in pairs_to_slice:
            try:
                start, end = f_map.get_indices(pair)

                # start/end are positions in the *unbinned* flat vector, where
                # each pair occupies a contiguous block of length n_ell:
                # pair_idx * n_ell to (pair_idx + 1) * n_ell. Recover which
                # block this is and the local (within-block) ell range.
                pair_idx = start // n_ell
                local_start = start - pair_idx * n_ell
                local_end = end - pair_idx * n_ell

                local_start_ell = l_min + local_start
                local_end_ell = l_min + local_end

                start_bin = ell_to_bin_index(local_start_ell, edges)
                end_bin = ell_to_bin_index(local_end_ell, edges)

                global_start = pair_idx * n_bins + start_bin
                global_end = pair_idx * n_bins + end_bin

                all_ranges.append(np.arange(global_start, global_end))
                final_sliced_pairs.append(pair)
            except ValueError as e:
                print(f"Warning: {e} Skipping this block from the slice.")
                continue

    if not all_ranges:
        raise ValueError("No matching spectra blocks were found to slice!")

    keep_indices = np.concatenate(all_ranges)

    full_matrix = cov_obj.matrix if hasattr(cov_obj, 'matrix') else cov_obj

    sliced_matrix_raw = full_matrix[keep_indices, :]
    sliced_matrix_raw = sliced_matrix_raw[:, keep_indices]

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
    def __init__(self, n_lens=4, n_src=4, l_min=2, n_ell=3000, desired_pairs=None, cmb_primaries=False):
        self.n_lens = n_lens
        self.n_src = n_src
        self.l_min = l_min
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
    def __init__(self, f_map, spectra_dict, f_sky_c, f_sky_g, f_sky_l, f_sky_c_g, f_sky_c_l, f_sky_g_l, f_sky_c_g_l, binsize=1, logarithmic=False):
        self.f_map = f_map # ForecastMap object
        self.spectra_dict = spectra_dict #dictionary mapping (tracer1, tracer2) to C_l^(tracer1, tracer2) array of length n_ell
        self.f_sky_c = f_sky_c 
        self.f_sky_g = f_sky_g
        self.f_sky_l = f_sky_l 
        self.f_sky_c_g = f_sky_c_g
        self.f_sky_c_l = f_sky_c_l
        self.f_sky_g_l = f_sky_g_l
        self.f_sky_c_g_l = f_sky_c_g_l
        self.binsize = binsize
        self.logarithmic = logarithmic
        self.l_min = f_map.l_min

        self.N_ell_unbinned = f_map.n_ell
        self.N_ell_binned = int(np.ceil(self.N_ell_unbinned / self.binsize))
        self.edges = get_ell_bin_edges(self.l_min, self.N_ell_unbinned, self.binsize, self.logarithmic)
        self.N_ell_binned = len(self.edges) - 1
        # total length of the flattened data vector after binning is
        self.N = len(f_map.pairs) * self.N_ell_binned

        # master covariance matrix dimensions are based on binned ell values
        self.matrix = np.zeros((self.N, self.N))
        self.block_slices = {}

        # build covariance
        self._build_master_covariance()

    # get correct f_sky 
    def get_f_sky(self, pair_A, pair_B):
        fields = list(pair_A) + list(pair_B)
        
        # Classify each tracer field into 'C' (CMB), 'G' (Lens Galaxies), or 'L' (Source Lensing)
        categories = set()
        for f in fields:
            # Check string prefixes/names based on your field naming scheme
            if f in ('T', 'E', 'kappa_c'):
                categories.add('C')
            elif f.startswith('g'):        # e.g., 'g1', 'g2' (lens galaxies)
                categories.add('G')
            elif f.startswith('kappa_g'):  # e.g., 'kappa_g1' (source galaxy lensing)
                categories.add('L')
            else:
                raise ValueError(f"Unknown field tracer: {f}")
    
        # Map the unique categories present in the 4-field set to the corresponding f_sky
        if categories == {'C'}:
            return self.f_sky_c
        if categories == {'G'}:
            return self.f_sky_g
        if categories == {'L'}:
            return self.f_sky_l
        elif categories == {'C', 'G'}:
            return self.f_sky_c_g
        elif categories == {'C', 'L'}:
            return self.f_sky_c_l
        elif categories == {'G', 'L'}:
            return self.f_sky_g_l
        elif categories == {'C', 'G', 'L'}:
            return self.f_sky_c_g_l
        else:
            raise ValueError(f"Unhandled tracer field combination: {categories}")
            
    def _compute_block(self, pair_A, pair_B):

        # ells from 2 to N_ell_unbinned + 1, so the indices i directly correspond to ell_values[i-2]
        ells_unbinned = np.arange(self.l_min, self.N_ell_unbinned + self.l_min)

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
                return np.zeros(self.N_ell_unbinned) 

        Cl_ac = get_Cl(a, c)
        Cl_bd = get_Cl(b, d)
        Cl_ad = get_Cl(a, d)
        Cl_bc = get_Cl(b, c)

        # initialize
        binned_block = np.zeros((self.N_ell_binned, self.N_ell_binned))

        # Calculate covariance for each original ell, then bin
        for i_bin in range(self.N_ell_binned):
            # Bin boundaries now come from self.edges (raw ell values), converted
            # to 0-indexed positions in the unbinned Cl arrays.
            start_idx_unbinned = self.edges[i_bin] - self.l_min
            end_idx_unbinned = self.edges[i_bin + 1] - self.l_min
            
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

            # Calculate the unbinned Gasussian cov terms for the diagonal elements within this bin
            # fsky depends on which covariance is being calculated
            weights = 2 * current_ells_for_bin + 1
            f_sky = self.get_f_sky(pair_A, pair_B)
            unbinned_var_ell = (current_Cl_ac * current_Cl_bd + current_Cl_ad * current_Cl_bc) / (weights * f_sky)
            binned_block[i_bin, i_bin] = np.sum((weights ** 2) * unbinned_var_ell) / (np.sum(weights) ** 2)

        return binned_block

    def _build_master_covariance(self):
        for pair_A in self.f_map.pairs:

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

# -------------------------------------------------------------------------------------------------------------------------------------------- #

# SO DESI Likelihood (w/ or w/o emulator, w or w/o primaries) 
###### FIX TO ACCOUNT FOR LOGARITHMIC BINNING
class SO_x_DESI_Likelihood(Likelihood):

    params = {
        "Omega_c": None, # cold dark matter density
        "A_s": None,     # amplitude of primordial fluctuations
        "h": None,       # Hubble parameter
        "Omega_b": None, # baryon density
        "n_s": None,     # primordial tilt
        "w0": None,      # dark energy equation of state parameter
        "wa": None,      # dark energy equation of state parameter evolution
        "Omega_k": None, # curvature density (for curved LCDM) - will set to 0 for flat_LCDM
        "Neff": None,    # effective number of massless neutrinos present -- defaults to 3.044
        "m_nu": None,    # mass in eV of the massive neutrinos present
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
        self.f_sky_c = self.data_specs.get('f_sky_c') 
        self.f_sky_g = self.data_specs.get('f_sky_g') 
        self.f_sky_l = self.data_specs.get('f_sky_l') 
        self.f_sky_c_g = self.data_specs.get('f_sky_c_g') 
        self.f_sky_g_l = self.data_specs.get('f_sky_g_l') 
        self.f_sky_c_g_l = self.data_specs.get('f_sky_c_g_l') 
        self.l_min = self.data_specs.get('l_min') # max unbinned ell
        self.n_ell = self.data_specs.get('n_ell') # max unbinned ell
        self.binsize = self.data_specs.get('binsize') # binning size for ell
        self.magnification_bias_lenses = self.data_specs.get('magnification_bias_lenses')
        self.z_max = self.data_specs.get('z_max')
        self.n_chi = self.data_specs.get('n_chi')
        self.cmb_primaries = self.data_specs.get('cmb_primaries')

        # set desired spectra
        desired_spectra = self.data_specs.get('desired_spectra')
        if desired_spectra != 'None':
            self.desired_spectra = desired_spectra
        elif self.cmb_primaries:
            self.desired_spectra = ['GG', 'LL', 'GL', 'CC', 'CL', 'CG', 'TT', 'TE', 'EE', 'GT', 'LT', 'CT', 'EG', 'EL', 'CE']
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
            
        # retrieve lens and source data arrays
        self.lens_data = np.load(self.data_specs['lens_data_path'])
        self.source_data = np.load(self.data_specs['source_data_path'])

        print(f"  Loaded lens data from {self.data_specs['lens_data_path']}")
        print(f"  Loaded source data from: {self.data_specs['source_data_path']}")

        # check for pre-computed data vector and covariance matrix paths
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
            self.f_map = ForecastMap(n_lens=num_lens_bins, n_src=num_source_bins, l_min=self.l_min, n_ell=self.n_ell, desired_pairs=desired_pairs, cmb_primaries = self.cmb_primaries)

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
                    f_sky_c = self.f_sky_c,
                    f_sky_g = self.f_sky_g,
                    f_sky_l = self.f_sky_l,
                    f_sky_c_g = self.f_sky_c_g,
                    f_sky_c_l = self.f_sky_c_l,
                    f_sky_g_l = self.f_sky_g_l,
                    f_sky_c_g_l = self.f_sky_c_g_l,
                    l_min=self.l_min,
                    n_ell=self.n_ell,
                    binsize=self.binsize,
                    shot_noise_lens=self.shot_noise_lens,
                    shape_noise_source=self.shape_noise_source,
                    cmb_noise_kk=self.cmb_noise_kk,
                    cmb_noise_TT=self.cmb_noise_TT,
                    cmb_noise_EE=self.cmb_noise_EE,
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

        Omega_c = kwargs['Omega_c']
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
        ells = np.arange(self.l_min, self.n_ell + self.l_min) # unbinned ells
        lens_tracers, source_tracers, cmb_lensing_tracer = build_tracers_from_data(
            current_cosmology, self.lens_data, self.source_data, self.magnification_bias_lenses)
        tracer_dict = build_tracer_dict(lens_tracers, source_tracers, cmb_lensing_tracer)
        noise_dict = build_noise_dict(self.f_map, ells, self.shot_noise_lens, self.shape_noise_source, cmb_noise_kk = self.cmb_noise_kk, cmb_noise_TT = self.cmb_noise_TT, cmb_noise_EE = self.cmb_noise_EE)
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
        Omega_c = kwargs.get('Omega_c')
        Omega_b = kwargs.get('Omega_b')
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
        ells = np.arange(self.l_min, self.n_ell + self.l_min)
        lens_tracers, source_tracers, cmb_lensing_tracer = build_tracers_from_data(
            current_cosmology, self.lens_data, self.source_data, self.magnification_bias_lenses)
        tracer_dict = build_tracer_dict(lens_tracers, source_tracers, cmb_lensing_tracer)
        noise_dict = build_noise_dict(self.f_map, ells, self.shot_noise_lens, self.shape_noise_source, cmb_noise_kk = self.cmb_noise_kk, cmb_noise_TT = self.cmb_noise_TT, cmb_noise_EE = self.cmb_noise_EE)
        current_spectra_dict = build_spectra_dict(current_cosmology, self.f_map, tracer_dict, ells, noise_dict, linear_emulator=self.linear_emulator, boost_emulator=self.boost_emulator, cmb_primaries = self.cmb_primaries)

        # flatten the current Cls into a model data vector 'M'
        model_data_vector = build_data_vector(self.f_map, current_spectra_dict, self.n_ell, self.binsize)

        # Calculate the residual vector (D - M)
        difference_vector = self.observed_data_vector - model_data_vector
        
        # Calculate Chi-squared: r^T * InvCov * r
        chi2 = difference_vector.dot(self.inv_covariance.dot(difference_vector))
        
        return chi2
        
# -------------------------------------------------------------------------------------------------------------------------------------------- #
##### FIX TO ACCOUNT FOR LOGARITHMIC BINNING
# ref https://arxiv.org/pdf/astro-ph/9706198
# Fisher Forecast class
class FisherForecaster:
    def __init__(self, cosmology, lens_data, source_data, f_sky_c=0.4, f_sky_g=None, f_sky_l=None, f_sky_c_g=None, f_sky_c_l=None, f_sky_g_l=None, 
                 f_sky_c_g_l=None, l_min = 2, n_ell=5000, binsize=100, logarithmic=False, 
                 shot_noise_lens=None, shape_noise_source=None, cmb_noise_kk=None, cmb_noise_TT=None, 
                 cmb_noise_EE=None, magnification_bias_lenses=None, desired_spectra=None, 
                 linear_emulator=None, boost_emulator=None, step_dict=None, cmb_primaries=False, z_max=6, n_chi=4096, 
                 additional_Fisher_matrix=None, additional_Fisher_params=None, baryon_feedback_model = None, log10_T_AGN = None):

        self.cosmology = cosmology
        self.lens_data = lens_data
        self.source_data = source_data
        self.cmb_primaries = cmb_primaries
        self.z_max = z_max
        self.n_chi = n_chi
        self.binsize = binsize
        self.logarithmic = logarithmic
        self.additional_Fisher_matrix=additional_Fisher_matrix
        self.additional_Fisher_params=additional_Fisher_params
        self.baryon_feedback_model = baryon_feedback_model

        if self.baryon_feedback_model == 'hmcode':
            self.log10_T_AGN = log10_T_AGN
    
        self.survey_params = {
            'f_sky_c': f_sky_c, 'f_sky_g': f_sky_g, 'f_sky_l': f_sky_l, 'f_sky_c_g': f_sky_c_g, 'f_sky_c_l': f_sky_c_l, 'f_sky_g_l': f_sky_g_l, 
            'f_sky_c_g_l': f_sky_c_g_l, 'l_min': l_min, 'n_ell': n_ell, 'binsize': binsize, 'logarithmic': logarithmic,
            'shot_noise_lens': shot_noise_lens, 'shape_noise_source': shape_noise_source,
            'cmb_noise_kk': cmb_noise_kk, 'cmb_noise_TT': cmb_noise_TT, 'cmb_noise_EE': cmb_noise_EE,
            'magnification_bias_lenses': magnification_bias_lenses, 'desired_spectra': desired_spectra, 
            'linear_emulator': linear_emulator, 'boost_emulator': boost_emulator, 'cmb_primaries': cmb_primaries, 
            'z_max': z_max, 'n_chi': n_chi
        }
        
        # step sizes for numerical derivatives
        self.step_dict = step_dict if step_dict is not None else {
            'Omega_m': 1e-4, 'A_s': 2e-11, 'h': 1e-3, 'w0': 1e-2, 'wa': 1e-2, 'n_s': 1e-3, 'Omega_b': 1e-4, 
            'Omega_c': 1e-4, 'Omega_k': 1e-3, 'Neff': 1e-2, 'm_nu': 1e-4, 'T_CMB': 1e-2, 'Omega_lambda': 1e-3 
            # generally have the step about 1% of the value
        }
        
        # extract and freeze our baseline fiducial truths
        self.fiducial_dict = self._extract_param_dict(self.cosmology)
        fiducial_logAs = np.log(1e10 * self.cosmology["A_s"])
        self.fiducial_dict["logA_s"] = fiducial_logAs

        # matrices initialized to None until computed
        self.F = None    # Fisher matrix
        self.cov = None  # parameter covariance matrix

        # build full f_map, etc. so that I don't have to remake them every time I call build_theory_vector
        p = self.survey_params
        self.ells = np.arange(p['l_min'], p['l_min'] + p['n_ell']) 
        self.ell_bin_edges = get_ell_bin_edges(p['l_min'], p['n_ell'], p['binsize'], p['logarithmic'])
        self.num_binned_ells = len(self.ell_bin_edges) - 1
        self.full_f_map = ForecastMap(n_lens=self.lens_data.shape[1]-1, n_src=self.source_data.shape[1]-1, l_min=p['l_min'], n_ell=p['n_ell'], cmb_primaries = self.cmb_primaries)

        # default to full spectra
        if p['desired_spectra'] is None: 
            if self.cmb_primaries:
                p['desired_spectra'] = ['GG', 'LL', 'GL', 'CC', 'CL', 'CG', 'TT', 'EE', 'ET', 'GT', 'LT', 'CT', 'EG', 'EL', 'CE']
            else: 
                p['desired_spectra'] = ['GG', 'LL', 'GL', 'CC', 'CL', 'CG']

        sliced_pairs = create_simplified_desired_pairs(self.lens_data.shape[1] - 1, self.source_data.shape[1] - 1, p['desired_spectra'])
        self.final_f_map = ForecastMap(n_lens=self.lens_data.shape[1]-1, n_src=self.source_data.shape[1]-1, l_min=p['l_min'], n_ell=p['n_ell'], desired_pairs=sliced_pairs, cmb_primaries = self.cmb_primaries)
        self.sliced_pairs = sliced_pairs
        
    # get parameter dictionary from a ccl cosmology (I usually pass cosmologies, not dictionaries)
    def _extract_param_dict(self, cosmology):
        h = cosmology['h']
        Omega_b = cosmology['Omega_b']
        Omega_c = cosmology['Omega_c']
        Omega_m = cosmology['Omega_m']
        Omega_lambda = 1 - cosmology['Omega_m'] - cosmology['Omega_k']
        A_s = cosmology['A_s']
        n_s = cosmology['n_s']
        w0 = cosmology['w0']
        wa = cosmology['wa']
        Omega_k = cosmology['Omega_k']
        Neff = cosmology['Neff']
        m_nu = cosmology['m_nu']
        T_CMB = cosmology['T_CMB']

        # If pyccl returns m_nu as an array/list, sum it up to get the scalar total mass
        if hasattr(m_nu, '__len__') or isinstance(m_nu, (list, np.ndarray)):
            m_nu = np.sum(m_nu)
            
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
    def build_theory_vector(self, cosmology, noiseless = False, silent=True):

        if not silent:
            print("Computing theory vector...")
    
        cosmology.compute_growth()
        p = self.survey_params
        
        lens_tracers, source_tracers, cmb_lensing_tracer = build_tracers_from_data(
            cosmology, self.lens_data, self.source_data, p['magnification_bias_lenses'], z_max = p['z_max'], n_chi=p['n_chi'])
        tracer_dict = build_tracer_dict(lens_tracers, source_tracers, cmb_lensing_tracer)

        # build noisy or noieless vector, as needed
        if noiseless: 
            noise_dict = build_noise_dict(self.full_f_map, self.ells, None, None, cmb_noise_kk = None, cmb_noise_TT = None, cmb_noise_EE = None)
        else: 
            noise_dict = build_noise_dict(self.full_f_map, self.ells, p['shot_noise_lens'], p['shape_noise_source'], cmb_noise_kk = p['cmb_noise_kk'], cmb_noise_TT = p['cmb_noise_TT'], cmb_noise_EE = p['cmb_noise_EE'])

        full_spectra_dict = build_spectra_dict(cosmology, self.full_f_map, tracer_dict, self.ells, noise_dict, linear_emulator=p['linear_emulator'], boost_emulator=p['boost_emulator'], cmb_primaries=self.cmb_primaries)
        
        if self.sliced_pairs is not None: 
            final_spectra_dict = {pair: full_spectra_dict[pair] for pair in full_spectra_dict if pair in self.sliced_pairs}
        else:
            final_spectra_dict = full_spectra_dict

        self.spectra_dict = final_spectra_dict
            
        model_data_vector = []
        edges = self.ell_bin_edges
        for pair in self.final_f_map.pairs:
            unbinned_cls = final_spectra_dict[pair]
            for i_bin in range(len(edges) - 1):
                start_idx = edges[i_bin] - p['l_min']
                end_idx = min(edges[i_bin + 1] - p['l_min'], len(unbinned_cls))
                if start_idx < end_idx:
                    bin_cls = unbinned_cls[start_idx:end_idx]
                    bin_ells = self.ells[start_idx:end_idx]
                    weights = 2 * bin_ells + 1

                    # Mode-weighted average: sum((2l+1) * C_l) / sum(2l+1)
                    weighted_mean = np.sum(weights * bin_cls) / np.sum(weights)
                    model_data_vector.append(weighted_mean)
                else:
                    model_data_vector.append(0.0)
        
        return np.array(model_data_vector)
        
    # manually compute derivatives
    # 5-point stencil
    def get_derivatives(self, desired_params):
        C_derivatives = {}
        mu_derivatives = {}
        p = self.survey_params
    
        # Determine parameter names to iterate over
        has_logAs = "logA_s" in desired_params or "ln_10_10_As" in desired_params
        logAs_key = "logA_s" if "logA_s" in desired_params else "ln_10_10_As"
    
        # Map requested params to keys present in fiducial_dict
        eval_params = [p if p not in ["logA_s", "ln_10_10_As"] else "A_s" for p in desired_params]
    
        for param in set(eval_params):
            step = self.step_dict.get(param)
            
            # Setup parameter variations for 5-point stencil
            params_up1   = self.fiducial_dict.copy()
            params_up2   = self.fiducial_dict.copy()
            params_down1 = self.fiducial_dict.copy()
            params_down2 = self.fiducial_dict.copy()

            params_up1[param]   += step
            params_up2[param]   += 2.0 * step
            params_down1[param] -= step
            params_down2[param] -= 2.0 * step

            cosmo_up1   = self._make_cosmo(params_up1)
            cosmo_up2   = self._make_cosmo(params_up2)
            cosmo_down1 = self._make_cosmo(params_down1)
            cosmo_down2 = self._make_cosmo(params_down2)

            # build noiseless theory vector derivatives (mu)
            mu_up1   = self.build_theory_vector(cosmo_up1, noiseless=True)
            mu_up2   = self.build_theory_vector(cosmo_up2, noiseless=True)
            mu_down1 = self.build_theory_vector(cosmo_down1, noiseless=True)
            mu_down2 = self.build_theory_vector(cosmo_down2, noiseless=True)

            mu_derivatives[param] = (-mu_up2 + 8.0 * mu_up1 - 8.0 * mu_down1 + mu_down2) / (12.0 * step)

            # Covariance matrix derivatives (C)
            cov_up1, _, _   = build_covariance_from_data(cosmo_up1, self.lens_data, self.source_data, **p)
            cov_up2, _, _   = build_covariance_from_data(cosmo_up2, self.lens_data, self.source_data, **p)
            cov_down1, _, _ = build_covariance_from_data(cosmo_down1, self.lens_data, self.source_data, **p)
            cov_down2, _, _ = build_covariance_from_data(cosmo_down2, self.lens_data, self.source_data, **p)

            C_derivatives[param] = (-cov_up2.matrix + 8.0 * cov_up1.matrix - 8.0 * cov_down1.matrix + cov_down2.matrix) / (12.0 * step)
    
        # Convert A_s derivative to logA_s derivative: d(f)/d(ln 10^10 A_s) = A_s * d(f)/dA_s
        if has_logAs and "A_s" in C_derivatives:
            print("Converting from derivatives wrt A_s to derivatives wrt log10^10A_s")
            fiducial_As = self.fiducial_dict["A_s"]
            
            C_derivatives[logAs_key] = C_derivatives["A_s"] * fiducial_As
            mu_derivatives[logAs_key] = mu_derivatives["A_s"] * fiducial_As
    
            if "A_s" not in desired_params:
                del C_derivatives["A_s"]
                del mu_derivatives["A_s"]
    
        return C_derivatives, mu_derivatives
    
    # pull cosmology-building out so both get_derivatives and this can use it
    def _make_cosmo(self, p_dict):

        camb_extra = {"dark_energy_model": "ppf", "AccuracyBoost": 3}
        mps = 'halofit'

        if self.baryon_feedback_model == 'hmcode':
            mps = 'camb'
            camb_extra['halofit_version'] = 'mead2020_feedback'
            camb_extra['HMCode_logT_AGN'] = p_dict.get('log10_T_AGN', self.log10_T_AGN)

        cosmo = ccl.Cosmology(
            Omega_c = p_dict['Omega_c'],
            Omega_b = p_dict['Omega_b'],
            Omega_k = p_dict['Omega_k'],
            h       = p_dict['h'],
            A_s     = p_dict['A_s'],
            n_s     = p_dict['n_s'],
            w0      = p_dict['w0'],
            wa      = p_dict['wa'],
            Neff    = p_dict['Neff'],
            m_nu    = p_dict['m_nu'],
            T_CMB   = p_dict['T_CMB'],
            transfer_function = 'boltzmann_camb',
            matter_power_spectrum = mps,
            extra_parameters={"camb": camb_extra}
        )
        cosmo.compute_growth()
        return cosmo

    #### CHECK
    def _chi2(self, theta_dict, mu_fiducial, inv_C):
        cosmo = self._make_cosmo(theta_dict)
        mu = self.build_theory_vector(cosmo)
        d = mu - mu_fiducial
        return d @ inv_C @ d

    def compute_baryon_feedback_bias(self, T_AGN_true, T_AGN_assumed=None, C=None, desired_params=None):
        # ref https://academic.oup.com/mnras/article/391/1/228/1120808
        """
        Linear Fisher-bias estimate of the shift induced on desired_params if the
        true sky has HMCode log10(T_AGN) = T_AGN_true but the analysis (F, C,
        mu_derivatives) was built assuming T_AGN_assumed (defaults to self.log10_T_AGN).
        Requires self.baryon_feedback_model == 'hmcode' and make_fisher_matrix()
        to have already been run (uses self.F, self.cov, self.mu_derivatives).
        """
        
        if self.baryon_feedback_model != 'hmcode':
            raise ValueError("compute_baryon_feedback_bias currently assumes baryon_feedback_model='hmcode'.")
        if self.F is None or self.cov is None or not hasattr(self, 'mu_derivatives'):
            raise ValueError("Run make_fisher_matrix() first so F, cov, and mu_derivatives are populated.")
    
        desired_params = desired_params if desired_params is not None else self.desired_params
        T_AGN_assumed = T_AGN_assumed if T_AGN_assumed is not None else self.log10_T_AGN
    
        p = self.survey_params
        if C is None:
            cov_obj, _, _ = build_covariance_from_data(self.cosmology, self.lens_data, self.source_data, **p)
            C = cov_obj.matrix
        inv_C = np.linalg.inv(C)
    
        # theory vector at the *assumed* feedback level (should match what F/mu_derivatives used)
        old_T_AGN = self.log10_T_AGN
        self.log10_T_AGN = T_AGN_assumed
        mu_assumed = self.build_theory_vector(self._make_cosmo(self.fiducial_dict), noiseless=True, silent=True)
    
        # theory vector at the *true* feedback level
        self.log10_T_AGN = T_AGN_true
        mu_true = self.build_theory_vector(self._make_cosmo(self.fiducial_dict), noiseless=True, silent=True)
    
        self.log10_T_AGN = old_T_AGN  # restore state
    
        delta_mu = mu_true - mu_assumed
    
        B = np.array([self.mu_derivatives[param] @ inv_C @ delta_mu for param in desired_params])
        delta_theta = self.cov @ B   # self.cov == F^{-1} from make_fisher_matrix
    
        bias_dict = dict(zip(desired_params, delta_theta))
        param_index = {p: i for i, p in enumerate(self.desired_params)}
        sigma_dict = {param: np.sqrt(self.cov[param_index[param], param_index[param]]) for param in desired_params}

        print("Bias from assuming log10(T_AGN) = {:.3f} when truth is {:.3f}:".format(T_AGN_assumed, T_AGN_true))
        for param in desired_params:
            ratio = bias_dict[param] / sigma_dict[param]
            flag = "  <-- >0.3σ shift" if abs(ratio) > 0.3 else ""
            print(f"  Δ{param} = {bias_dict[param]:.4e}  ({ratio:+.2f}σ){flag}")
    
        return bias_dict
    
    #### CHECK
    # find optimal step sizes for each paramter, using our 5 point stencil approximation of the derivative
    # we want a step size that is in a stable plateau, so small variations are not drastically changing constraints 
    # -- if they are, this suggests were in an unstable regime and are not getting true derivatives, but noise artifacts, etc.
    def find_optimal_steps_5point(self, params=None, C=None, mu_fiducial=None,
                                   n_steps=30, h_min_frac=1e-6, h_max_frac=1e-1,
                                   window_frac=0.3, plot=True, verbose=True):

        if params is None:
            params = ['Omega_c', 'A_s', 'h', 'w0', 'wa', 'n_s', 'Omega_b',
                       'Omega_k']

        p = self.survey_params
        if C is None:
            cov_obj, _, _ = build_covariance_from_data(
                self.cosmology, self.lens_data, self.source_data, **p)
            C = cov_obj.matrix
        inv_C = np.linalg.inv(C)

        if mu_fiducial is None:
            mu_fiducial = self.build_theory_vector(self.cosmology, silent=True)

        chi2_center = self._chi2(self.fiducial_dict, mu_fiducial, inv_C)

        results = {}
        window_size = max(3, int(round(n_steps * window_frac)))

        for param in params:
            fid_val = self.fiducial_dict.get(param, None)
            if not fid_val:
                if verbose:
                    print(f"[{param}] skipped: zero/missing fiducial value.")
                continue

            # h_max_frac must leave room for 2h to stay inside a sane range
            h_values = np.logspace(np.log10(h_min_frac), np.log10(h_max_frac), n_steps) * abs(fid_val)
            F_pp = np.empty(n_steps)

            for k, h in enumerate(h_values):
                # 5-point stencil applied directly to chi2's *gradient* isn't quite
                # right (chi2 isn't mu), so instead build mu-derivative via the
                # real 5-point stencil, then form F_pp = dmu^T inv_C dmu, which is
                # exactly what make_fisher_matrix computes for this diagonal entry.
                theta_up1 = self.fiducial_dict.copy(); theta_up1[param] += h
                theta_up2 = self.fiducial_dict.copy(); theta_up2[param] += 2*h
                theta_dn1 = self.fiducial_dict.copy(); theta_dn1[param] -= h
                theta_dn2 = self.fiducial_dict.copy(); theta_dn2[param] -= 2*h

                mu_up1 = self.build_theory_vector(self._make_cosmo(theta_up1), silent=True)
                mu_up2 = self.build_theory_vector(self._make_cosmo(theta_up2), silent=True)
                mu_dn1 = self.build_theory_vector(self._make_cosmo(theta_dn1), silent=True)
                mu_dn2 = self.build_theory_vector(self._make_cosmo(theta_dn2), silent=True)

                dmu = (-mu_up2 + 8*mu_up1 - 8*mu_dn1 + mu_dn2) / (12.0 * h)
                F_pp[k] = dmu @ inv_C @ dmu

            best_start, best_rel_std = 0, np.inf
            for start in range(0, n_steps - window_size + 1):
                window = F_pp[start:start + window_size]
                w_mean = np.mean(window)
                if w_mean == 0:
                    continue
                rel_std = np.std(window) / abs(w_mean)
                if rel_std < best_rel_std:
                    best_rel_std = rel_std
                    best_start = start

            plateau_mask = np.zeros(n_steps, dtype=bool)
            plateau_mask[best_start:best_start + window_size] = True
            center_idx = best_start + window_size // 2
            recommended_h = h_values[center_idx]

            results[param] = {
                'h_values': h_values, 'F_pp': F_pp, 'plateau_mask': plateau_mask,
                'recommended_h': recommended_h,
                'plateau_value': np.mean(F_pp[plateau_mask]),
                'plateau_rel_std': best_rel_std,
            }
            if verbose:
                flag = "  <-- WARNING: scatter >2%" if best_rel_std > 0.02 else ""
                print(f"[{param}] recommended h = {recommended_h:.3e}  "
                      f"(F_pp = {results[param]['plateau_value']:.4e}, "
                      f"scatter = {best_rel_std:.2%}){flag}")

        if plot:
            ncols = 3
            nrows = int(np.ceil(len(results) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows), squeeze=False)
            for idx, (param, r) in enumerate(results.items()):
                ax = axes[idx // ncols][idx % ncols]
                ax.plot(r['h_values'], r['F_pp'], marker='o', ms=4, color='steelblue')
                ax.plot(r['h_values'][r['plateau_mask']], r['F_pp'][r['plateau_mask']],
                         marker='o', ms=6, color='firebrick', lw=2)
                ax.axvline(r['recommended_h'], color='firebrick', ls='--')
                ax.set_xscale('log')
                ax.set_title(f"{param} (h*={r['recommended_h']:.2e})", fontsize=9)
            for idx in range(len(results), nrows*ncols):
                axes[idx//ncols][idx % ncols].axis('off')
            plt.tight_layout(); plt.show()

        return results

    # plot derivatives of spectra wrt different parameters
    ##### CHECK
    def plot_derivatives(self, desired_params=None, normalized=False):
        
        if not hasattr(self, 'mu_derivatives') or self.mu_derivatives is None:
            raise AttributeError(
                "mu_derivatives not found. Please run self.make_fisher_matrix() first."
            )
            
        # Survey parameters and points per pair
        p = self.survey_params
        n_points_per_pair = self.num_binned_ells
        edges = self.ell_bin_edges

        # x-axis values (mode-weighted bin centers, consistent with build_theory_vector)
        x_data = []
        for i_bin in range(len(edges) - 1):
            start_idx = edges[i_bin] - p['l_min']
            end_idx = edges[i_bin + 1] - p['l_min']
            bin_ells = self.ells[start_idx:end_idx]
            weights = 2 * bin_ells + 1
            x_data.append(np.sum(weights * bin_ells) / np.sum(weights))
        x_data = np.array(x_data)

        # Extract ordered spectrum pairs
        ordered_pairs = self.final_f_map.pairs 

        # Retrieve the spectra dictionary
        spectra_dict = getattr(self, 'final_spectra_dict', getattr(self, 'spectra_dict', None))
        if normalized and spectra_dict is None:
            raise AttributeError("Could not find 'final_spectra_dict' or 'spectra_dict' to normalize by C_l.")

        for idx, pair in enumerate(ordered_pairs):
            fig, ax = plt.subplots(figsize=(8, 5))
            
            start_idx = idx * n_points_per_pair
            end_idx = start_idx + n_points_per_pair
            
            # Compute binned C_ell for this specific pair, mode-weighted and edge-consistent
            if normalized:
                unbinned_cl = np.array(spectra_dict[pair])
                c_ell_segment = []
                for i_bin in range(len(edges) - 1):
                    b_start = edges[i_bin] - p['l_min']
                    b_end = edges[i_bin + 1] - p['l_min']
                    bin_cls = unbinned_cl[b_start:b_end]
                    bin_ells = self.ells[b_start:b_end]
                    weights = 2 * bin_ells + 1
                    c_ell_segment.append(np.sum(weights * bin_cls) / np.sum(weights))
                c_ell_segment = np.array(c_ell_segment)

            for param in desired_params:
                if param not in self.mu_derivatives:
                    continue
                
                param_deriv_segment = self.mu_derivatives[param][start_idx:end_idx]

                if normalized:
                    # Both arrays are length num_binned_ells now
                    y_data = np.where(c_ell_segment != 0, param_deriv_segment / c_ell_segment, 0.0)
                    y_label = rf"$\frac{{1}}{{C_\ell^{{\text{{{pair}}}}}}} \frac{{\partial C_\ell^{{\text{{{pair}}}}}}}{{\partial {param}}}$"
                else:
                    y_data = param_deriv_segment
                    y_label = rf"$\frac{{\partial C_\ell^{{\text{{{pair}}}}}}}{{\partial {param}}}$"

                ax.plot(x_data, y_data, label=y_label, lw=2)

            ax.set_xlabel(r"Multipole $\ell$", fontsize=12)
            ax.set_ylabel(
                rf"$\frac{{1}}{{C_\ell}} \frac{{\partial \mu}}{{\partial \theta}}$" if normalized else rf"$\frac{{\partial \mu}}{{\partial \theta}}$", 
                fontsize=12
            )
            ax.set_title(f"Derivatives for Spectrum Pair: {pair}", fontsize=14)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="best", frameon=True)
            plt.tight_layout()
            plt.show()
            
    # build and invert Fisher matrix without considering priors, since they are uniform, not Gaussian
    def make_fisher_matrix(self, desired_params=None, C=None, mu=None, C_derivatives=None, mu_derivatives=None, print_summary = False):
        
        # default to full Fisher matrix
        if desired_params is None:
            desired_params = ['Omega_c', 'A_s', 'h', 'w0', 'wa', 'n_s', 'Omega_b', 'Omega_k', 'Neff', 'm_nu', 'T_CMB']
        
        self.desired_params = desired_params
        
        p = self.survey_params
        if self.additional_Fisher_params is not None and self.desired_params != self.additional_Fisher_params:
            print("WARNING: the built Fisher matrix and the additional Fisher matrix DO NOT HAVE THE SAME PARAMETERS.")
            
        # build fiducial covariance and theory vector if they are missing
        if C is None:
            cov_obj, _, _ = build_covariance_from_data(self.cosmology, self.lens_data, self.source_data, **p)
            C = cov_obj.matrix
        if mu is None:
            mu = self.build_theory_vector(self.cosmology)
            
#       if C_derivatives is None or mu_derivatives is None:
        C_derivatives, mu_derivatives = self.get_derivatives(desired_params)

        self.C_derivatives = C_derivatives
        self.mu_derivatives = mu_derivatives
        
        n_params = len(desired_params)
        F = np.zeros((n_params, n_params))
        inv_C = np.linalg.inv(C)
        F_mu = np.zeros((n_params, n_params))
        F_cov_only = np.zeros((n_params, n_params))

        #### WRITE OUT MATH
        for i, p_i in enumerate(desired_params):
            for j, p_j in enumerate(desired_params):
                dC_di = C_derivatives[p_i]
                dC_dj = C_derivatives[p_j]
                dmu_di = mu_derivatives[p_i][:, np.newaxis]
                dmu_dj = mu_derivatives[p_j][:, np.newaxis]
        
                matrix1 = inv_C @ dC_di @ inv_C @ dC_dj
                matrix2 = dmu_di.T @ inv_C @ dmu_dj
                #matrix2 = inv_C @ ((dmu_di @ dmu_dj.T) + (dmu_dj @ dmu_di.T))
        
                F_cov_only[i, j] = 0.5 * np.trace(matrix1)
                F_mu[i, j]  = matrix2.item()
                F[i, j] = F_cov_only[i,j] + F_mu[i, j]
                #F[i, j] = 0.5 * np.trace(matrix1 + matrix2)
                
        self.F = F

        # calculate covariance matrix
        self.cov = np.linalg.inv(self.F)   
        
        if print_summary:
            print("")
            print("Fisher Forecast Results Without Priors")
            for i, p_name in enumerate(self.desired_params):
                sigma = np.sqrt(self.cov[i, i])
                error_str = f"{sigma:.3e}" if sigma < 0.001 else f"{sigma:.4f}"
                print(f"The uncertainty on {p_name} is {error_str}")
            print("")

        return self.F, self.cov

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
    # if we want to plot paramaters that were not sampled over -- e.g. omega_m we'll need to convert
    def plot_with_cobaya_overlay(
        self, 
        title="Fisher Forecast vs. Cobaya MCMC Constraints",
        cobaya_chain_dir=None, 
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

        # fall back to plotting the parameters sampled over
        if plot_params is None:
            plot_params = self.desired_params
            
        raw_datasets = []
        legend_labels = []
        contour_colors = []

        latex_labels = {'Omega_m': r'\Omega_\mathrm{m}',
                        'Omega_b': r'\Omega_\mathrm{b}',
                        'Omega_k': r'\Omega_\mathrm{k}',
                        'Omega_c': r'\Omega_\mathrm{c}',
                        'Omega_lambda': r'\Omega_\mathrm{\Lambda}',
                        'wa': r'w_a',
                        'w0': r'w_0',
                        'h': r'h',
                        'A_s': r'A_\mathrm{s}',
                        'logA_s': r'logA_\mathrm{s}',
                        'n_s': r'n_\mathrm{s}',
                        'Neff': r'N_\mathrm{eff}',
                        'm_nu': r'm_\mathrm{nu}',
                        'T_CMB': r'T_\mathrm{CMB}'
                    }
        labels = [latex_labels.get(p, p) for p in plot_params]

        # Get main Fisher results
        fisher_dataset = self.sample_fisher_with_uniform_priors(uniform_priors=uniform_priors, num_samples=num_samples)
        raw_datasets.append(fisher_dataset)
        legend_labels.append("Fisher Forecast (With Priors)" if uniform_priors else "Fisher Forecast")
        contour_colors.append("firebrick")

        # Optional: overlay the prior-less Fisher distribution (if the main Fisher is not prior-less)
        if overlay_priorless and uniform_priors is not None:
            fisher_priorless = self.sample_fisher_with_uniform_priors(uniform_priors=None, num_samples=num_samples)
            raw_datasets.append(fisher_priorless)
            legend_labels.append("Fisher Forecast (No Priors)")
            contour_colors.append("gray")
        
        # Optional: parse and load Cobaya chains if directory is given
        if cobaya_chain_dir is not None:
                
            all_weights = []
            all_loglikes = []
            param_tracks = {p: [] for p in plot_params}
            
            for i in range(num_chains):
                chain_path = os.path.join(cobaya_chain_dir, f"chain_task_{i}.txt")
                if os.path.exists(chain_path):
                    data = np.loadtxt(chain_path)
                    burn = int(burn_in_fraction * len(data))
                    
                    all_weights.append(data[burn:, 0])
                    all_loglikes.append(data[burn:, 1])
                    for idx, param in enumerate(plot_params):
                        col_idx = idx + 2  # Skip weight and loglike columns
                        param_tracks[param].append(data[burn:, col_idx])
                else:
                    print(f"Warning: {chain_path} not found. Skipping.")
            
            if len(all_weights) == 0:
                raise FileNotFoundError(f"No valid chain files found in {cobaya_chain_dir}")
                
            combined_samples = np.column_stack([np.concatenate(param_tracks[p]) for p in plot_params])
            cobaya_labels = [latex_labels.get(p, p) for p in plot_params]
            
            mcmc_samples = MCSamples(
                samples=combined_samples,
                weights=np.concatenate(all_weights),
                loglikes=np.concatenate(all_loglikes),
                names=self.desired_params,
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
            for p in self.desired_params:
                if p in uniform_priors:
                    p_min, p_max = uniform_priors[p]
                    prior_samples_dict[p] = np.random.uniform(p_min, p_max, size=num_samples)
                else:
                    #### WHAT DO I DO IF ITS NOT IN UNIFORM_PRIORS -- I DON'T WANT TO SHOW PRIORS WHERE THEY DON'T EXIST
                    fiducial = float(self.fiducial_dict[p])
                    prior_samples_dict[p] = np.random.uniform(fiducial * 0.5, fiducial * 1.5, size=num_samples)
                    
            prior_df = pd.DataFrame(prior_samples_dict)
            
            prior_dataset = MCSamples(
                samples=prior_df.values, 
                names=self.desired_params, 
                labels=labels, 
                name_tag="Uniform Priors"
            )
            
            # Insert it at the BEGINNING of your lists so it renders in the background
            raw_datasets.insert(0, prior_dataset)
            legend_labels.insert(0, "Uniform Priors")
            contour_colors.insert(0, "lightgray") 
        
        # Create GetDist Subplot Plotter and generate the grid
        n_params = len(plot_params)
        g = plots.get_subplot_plotter(width_inch=2.5 * n_params)
        
        # Extract fiducial values in correct order for markers
        fiducial_vals = {p: float(self.fiducial_dict[p]) for p in plot_params if p in self.fiducial_dict}

        # Loop through each dataset and update the parameter labels manually
        for dataset in raw_datasets:
            for param_name, latex_string in latex_labels.items():
                # Get the list of names and check using 'in'
                if param_name in [p.name for p in dataset.paramNames.names]:
                    dataset.paramNames.parWithName(param_name).label = latex_string

        # loop through datasets and convert from self.desired_params to plot_params
        plot_datasets = []
        raw_idx = {param: idx for idx, param in enumerate(self.desired_params)}
        
        for dataset in raw_datasets:
            samples = dataset.samples
            # Perform the column transformation if bases differ
            if list(self.desired_params) != list(plot_params):
                projected_columns = []
                for p in plot_params:
                    chain_length = samples.shape[0]                    
                    if p == 'Omega_m':
                        b_col = raw_idx.get('Omega_b')
                        c_col = raw_idx.get('Omega_c')
                        m_nu_col = raw_idx.get('m_nu')
                        h_col = raw_idx.get('h')

                        b_samples = samples[:, b_col] if b_col is not None else np.full(chain_length, self.cosmology['Omega_b'])
                        c_samples = samples[:, c_col] if c_col is not None else np.full(chain_length, self.cosmology['Omega_c'])
                        m_nu_samples = samples[:, m_nu_col] if m_nu_col is not None else np.full(chain_length, np.sum(self.cosmology['m_nu']))
                        h_samples = samples[:, h_col] if h_col is not None else np.full(chain_length, self.cosmology['h'])
                        Omega_nu_samples = m_nu_samples / (h_samples * h_samples * 93.15) #### CHECK

                        projected_columns.append(b_samples + c_samples + Omega_nu_samples)
                        
                    elif p == 'Omega_lambda':
                        b_col = raw_idx.get('Omega_b')
                        c_col = raw_idx.get('Omega_c')
                        k_col = raw_idx.get('Omega_k')
                        m_nu_col = raw_idx.get('m_nu')
                        h_col = raw_idx.get('h')
                        
                        b_samples = samples[:, b_col] if b_col is not None else np.full(chain_length, self.cosmology['Omega_b'])
                        c_samples = samples[:, c_col] if c_col is not None else np.full(chain_length, self.cosmology['Omega_c'])
                        k_samples = samples[:, k_col] if k_col is not None else np.full(chain_length, self.cosmology['Omega_k'])
                        m_nu_samples = samples[:, m_nu_col] if m_nu_col is not None else np.full(chain_length, np.sum(self.cosmology['m_nu']))
                        h_samples = samples[:, h_col] if h_col is not None else np.full(chain_length, self.cosmology['h'])
                        Omega_nu_samples = m_nu_samples / (h_samples * h_samples * 93.15) #### CHECK
                        
                        projected_columns.append(1.0 - b_samples - c_samples - k_samples - Omega_nu_samples)
                    else:
                        projected_columns.append(samples[:, raw_idx[p]])
                        
                samples = np.column_stack(projected_columns)

            # Package up into GetDist format
            mcsamples = MCSamples(
                samples=samples, 
                names=plot_params, 
                labels=labels,
                name_tag=dataset.name_tag
            )
        
            plot_datasets.append(mcsamples)
        
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

        # Quantitative Parameter Comparison (Table Output)
        if print_summary:
            # Initialize the single master table header
            md_lines = [
                "| Parameter | Dataset / Model | 1-Sigma (68%) | 2-Sigma (95%) |",
                "| :--- | :--- | :---: | :---: |"
            ]
            
            for param in plot_params:
                display_label = latex_label_map.get(param, param)
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

                for dataset, label in zip(plot_datasets, legend_labels):
                    
                    val_1sig = dataset.getInlineLatex(param, limit=1)
                    val_2sig = dataset.getInlineLatex(param, limit=2)
                    
                    # If GetDist included an '=', split it to throw away its broken label (e.g., 'Omegam')
                    if "=" in val_1sig:
                        val_1sig = val_1sig.split("=")[-1].strip()
                    if "=" in val_2sig:
                        val_2sig = val_2sig.split("=")[-1].strip()
                    
                    # Rebuild the string using latex_label_map entry
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

#### CHECK
#Fisher forecasting using lensing ratio method
#Idea: For each lens-redshift bin i and ell bin, give the CMB-lensing x galaxy-density
#spectrum a free amplitude: C_ell^{(g_i, kappa_c)} = A_{i,ell}
#and predict EVERY paired galaxy-lensing x galaxy-density spectrum -- one per
#source bin j you choose to pair with lens bin i -- as a geometric multiple of
#that SAME shared amplitude: C_ell^{(g_i, kappa_g_j)} = r_{i,j,ell}(theta_geo) * A_{i,ell}
#for each j in lens_to_source_bin[i]. A given lens bin can be paired with one
#source bin or several; C_ell^{(g_i,kappa_c)} is a single physical measurement,
#so all of its paired GL spectra share the one amplitude -- pairing against
#more source bins adds more constraints on the SAME A, not more free A's.
#r is computed from lensing-kernel integrals with P(k,z) FROZEN at its
#fiducial shape -- only background distances move with theta_geo. theta_geo is
#whichever subset of {Omega_m, Omega_k, w0, wa} you have leverage on; growth/
#amplitude parameters (A_s, sigma8, bias, etc.) never enter -- fully absorbed
#into the free A_{i,ell}'s.
#All A_{i,ell} are marginalized with a flat prior the standard Fisher way:
#include them as ordinary parameters in the joint Fisher matrix, invert, and
#keep the theta_geo submatrix (Schur complement).

#Built against the actual ForecastMap/CovarianceMatrix/create_simplified_desired_pairs
#code: 'GL' expands to the full lens-bin x source-bin cross product (any i,j),
#'CG' to every lens bin, and final_f_map.pair_to_index gives the exact block
#position build_theory_vector's concatenation uses -- so index lookups here
#are direct, not guesswork.

### HOW DO I ADD PRIMARIES -- I WANT TO JUST ADD THEM TO THE DATA VECTOR SO YOU HAVE TO FIT TO BOTH
#### DOES THIS ASSUME R IS CONSTANT ACROSS ELL VALUES?
class LRFisherForecaster(FisherForecaster):

    def __init__(self, *args, lens_to_source_bin, geo_params=('Omega_k',),
             fiducial_pmm_cosmology=None, **kwargs):
        """
        lens_to_source_bin : dict {lens_bin_index (1-based): source_bin_index
            OR list of source_bin_indices} Which shear/source sample(s) to pair with each lens bin for the
            GL side of the ratio, e.g. {1: 3, 2: [3, 4], 3: 4}. Required
        geo_params : tuple of str
            Which background-cosmology parameters make up theta_geo, e.g.
            ('Omega_k',) or ('w0','wa') with Omega_m fixed. H0/h does not
            belong here -- it cancels in r by construction.
        fiducial_pmm_cosmology : pyccl.Cosmology or None
            Cosmology used ONLY to freeze P(k,z) for the r calculation.
            Defaults to self.cosmology. Never re-derived per theta_geo trial.
        """
        super().__init__(*args, **kwargs)
        self.geo_params = list(geo_params)
        self.lens_to_source_bin = lens_to_source_bin

        self._fiducial_pmm_cosmo = fiducial_pmm_cosmology or self.cosmology
        self._pk_frozen = self._freeze_pk(self._fiducial_pmm_cosmo)

        # cg_gl_map: one "amplitude group" per (lens_bin, ell_bin), each
        # holding its single CG entry and a LIST of GL entries (one per
        # paired source bin) that all share that group's amplitude.
        self.cg_gl_map = self._build_cg_gl_map(lens_to_source_bin)

        # Flat list of every individual GL entry across all groups, in a
        # fixed order used consistently by compute_r_vector, _design_matrix,
        # and the theta_geo derivative loop 
        self._gl_flat = []
        for g_idx, group in enumerate(self.cg_gl_map):
            for gl in group['gl_entries']:
                # tie each kappa_g*g back to its corresponding kappa_c*g 
                self._gl_flat.append({
                    'group_idx': g_idx,
                    'ell_bin': group['ell_bin'],
                    'source_bin': gl['source_bin'],
                    'gl_pair': gl['gl_pair'],
                    'gl_idx': gl['gl_idx'],
                })

    # find and freeze P(k, z) so it doesn't depend on trial theta_geo
    # when we calculate ratios in the future we will forcibly keep P(k, z) frozen, so that even if it SHOULD have an effect it's not allowed to 
    # this speeds things up
    #### will this sway things at all?
    def _freeze_pk(self, cosmology):
        cosmology.compute_growth()
        return cosmology.get_nonlin_power()

    # build map of spectra pairs
    @staticmethod
    def _canonical_pair(a, b):
        # matches ForecastMap's own canonicalization exactly (plain string compare)
        return (a, b) if a <= b else (b, a)

    def _build_cg_gl_map(self, lens_to_source_bin):
        cg_gl_map = []
        for i, j_or_js in lens_to_source_bin.items():
            source_bins = list(j_or_js) if isinstance(j_or_js, (list, tuple)) else [j_or_js]

            cg_pair = self._canonical_pair(f'g{i}', 'kappa_c')
            if cg_pair not in self.final_f_map.pair_to_index:
                raise ValueError(
                    f"{cg_pair} not found in final_f_map.pairs -- make sure "
                    f"desired_spectra includes 'CG'."
                )
            idx_cg = self.final_f_map.pair_to_index[cg_pair]

            gl_lookups = []
            for j in source_bins:
                gl_pair = self._canonical_pair(f'g{i}', f'kappa_g{j}')
                if gl_pair not in self.final_f_map.pair_to_index:
                    raise ValueError(
                        f"{gl_pair} not found in final_f_map.pairs -- make "
                        f"sure desired_spectra includes 'GL'."
                    )
                gl_lookups.append((j, gl_pair, self.final_f_map.pair_to_index[gl_pair]))

            for ell_bin in range(self.num_binned_ells):
                cg_gl_map.append({
                    'lens_bin': i,
                    'ell_bin': ell_bin,
                    'cg_pair': cg_pair,
                    'cg_idx': idx_cg * self.num_binned_ells + ell_bin,
                    'gl_entries': [
                        {
                            'source_bin': j,
                            'gl_pair': gl_pair,
                            'gl_idx': idx_gl * self.num_binned_ells + ell_bin,
                        }
                        for (j, gl_pair, idx_gl) in gl_lookups
                    ],
                })
        return cg_gl_map

    # compute the weighted average for a spectra for one of its multipole bins
    def _bin_one_ell(self, unbinned_cls, ell_bin_idx):
        """Mode-weighted binning for a single ell bin, matching the parent
        class's build_theory_vector convention exactly."""
        p = self.survey_params
        edges = self.ell_bin_edges
        start_idx = edges[ell_bin_idx] - p['l_min']
        end_idx = min(edges[ell_bin_idx + 1] - p['l_min'], len(unbinned_cls))
        bin_cls = unbinned_cls[start_idx:end_idx]
        bin_ells = self.ells[start_idx:end_idx]
        weights = 2 * bin_ells + 1
        return np.sum(weights * bin_cls) / np.sum(weights)

    # compute ratio for a given cosmology, for all the spectra involved
    def compute_r_vector(self, cosmology):
        cosmology.compute_growth()

        lens_tracers, source_tracers, cmb_tracer = build_tracers_from_data(
            cosmology, self.lens_data, self.source_data,
            self.survey_params['magnification_bias_lenses'],
            z_max=self.survey_params['z_max'], n_chi=self.survey_params['n_chi'])
        tracer_dict = build_tracer_dict(lens_tracers, source_tracers, cmb_tracer)

        noiseless_noise_dict = build_noise_dict(
            self.full_f_map, self.ells, None, None,
            cmb_noise_kk=None, cmb_noise_TT=None, cmb_noise_EE=None)

        full_spectra = build_spectra_dict(
            cosmology, self.full_f_map, tracer_dict, self.ells,
            noiseless_noise_dict,
            linear_emulator=self.survey_params['linear_emulator'],
            boost_emulator=self.survey_params['boost_emulator'],
            cmb_primaries=False,
            pk_override=self._pk_frozen,
        )

        # find the Cl^kappa_g*g spectrum for a given bin of multipoles, put all together into an array
        # goes through each group (each Cl^kappa_c*g) -- e.g. array of arrays
        cg_vals = np.array([
            self._bin_one_ell(full_spectra[g['cg_pair']], g['ell_bin'])
            for g in self.cg_gl_map
        ])

        # compute Cl^kappa_c*g/Cl^kappa_g*g for each bin
        # divide Cl^kappa_c*g corresponding the the group that each kappa_c*c is in by that kappa_c*g 
        # one Cl^kappa_c*g may be divided by many different Cl^kappa_g*g because each lens galaxy bin can pair with multiple different source galaxy bins, but only with the one CMB
        # loop through ell bins in a group, then loop through groups
        r = np.zeros(len(self._gl_flat))
        for k, gl in enumerate(self._gl_flat):
            gl_val = self._bin_one_ell(full_spectra[gl['gl_pair']], gl['ell_bin'])
            r[k] = cg_vals[gl['group_idx']] / gl_val
        return r

    # finds amplitude of Cl^kappa_c*g (A_{i,ell}) for each group
    # this is fiducial, and is NOT P(k) indep
    def compute_fiducial_amplitudes(self):
        mu_fid = self.build_theory_vector(self.cosmology, noiseless=True)
        return np.array([mu_fid[g['cg_idx']] for g in self.cg_gl_map])

    # Analytic design matrix (dmu/dA): exact, no finite differencing.
    # X[i, j] gives the derivative of the i^th spectra wrt the j^th amplitude
    # Columns = amplitude groups; each CG row gets a 1 in its own
    # group's column, each GL row gets r in ITS group's column (so
    # multiple GL rows can point at the same column/amplitude).
    # n_total is the number of spectra times the number of ell values post-binning
    def _design_matrix(self, r_vector, n_total):
        n_amp = len(self.cg_gl_map) # number of Cl^kappa_c*g included = number of groupings
        X = np.zeros((n_total, n_amp))
        # put a 1 in the Cl^k_c*g row in that g's column
        for g_idx, group in enumerate(self.cg_gl_map):
            X[group['cg_idx'], g_idx] = 1.0
        # put r in the Cl^k_g*g row in it's g's column
        # r is dependent on the sources/lenses and the ell bin
        for k, gl in enumerate(self._gl_flat):
            X[gl['gl_idx'], gl['group_idx']] = r_vector[k]
        return X

    #### CHECK
    ## smoothness condiiton for A?
    ## splines?
    # 6. Joint Fisher matrix over theta_geo + all A_{i,ell}; marginalize A
    #    via the Schur complement (top-left block of the inverted joint matrix)
    def make_lr_fisher_matrix(self, C=None, finite_diff_step=None, print_summary=False):
        p = self.survey_params

        # compute fiducial covariance matrix, ratio, amplitude, and derivatives of vectors wrt amps
        if C is None:
            cov_obj, _, _ = build_covariance_from_data(
                self.cosmology, self.lens_data, self.source_data, **p)
            C = cov_obj.matrix
        inv_C = np.linalg.inv(C)
        n_total = C.shape[0]
        r_fid = self.compute_r_vector(self.cosmology)
        A_fid = self.compute_fiducial_amplitudes()
        X_fid = self._design_matrix(r_fid, n_total)

        # determine the number of parameters being constrained
        n_amp = len(self.cg_gl_map)
        n_geo = len(self.geo_params)
        n_par = n_geo + n_amp

        # dmu/d(theta_geo): nonzero only through r, only on GL rows
        # mu = [A, rA]
        # dmu/d(theta_geo) = [0, Adr]
        # this is the simplest derivative approx
        ### Should we go to 5pt stencil?
        # dmu_dtheta is a list of lists
        # dmu_dtheta[i[j]] = derivative of j'th spectra block in mu (certain ell bin, certain spectrum) wrt i'th parameter
        ### right now each A is set by the cosmology, but doesn't factor into the final fisher constraints
        ### in an MCMC how would we handle A? there we're fitting vectors explicitly so we can't have A have meaning...
        dmu_dtheta = {}
        for param in self.geo_params:
            h = finite_diff_step or self.step_dict.get(param)
            up = dict(self.fiducial_dict); up[param] = self.fiducial_dict[param] + h
            dn = dict(self.fiducial_dict); dn[param] = self.fiducial_dict[param] - h
            r_up = self.compute_r_vector(self._make_cosmo(up))
            r_dn = self.compute_r_vector(self._make_cosmo(dn))
            dr = (r_up - r_dn) / (2 * h)

            vec = np.zeros(n_total)
            for k, gl in enumerate(self._gl_flat):
                vec[gl['gl_idx']] = dr[k] * A_fid[gl['group_idx']]
            dmu_dtheta[param] = vec

        # assemble all derivatives, first those wrt parameters, then those wrt free amplitudes
        all_derivs = [dmu_dtheta[pname] for pname in self.geo_params]
        all_derivs += [X_fid[:, k] for k in range(n_amp)]

        # build Fisher
        ### right now this is only the part of the equation
        F = np.zeros((n_par, n_par))
        for a in range(n_par):
            for b in range(a, n_par):

                #matrix1 = inv_C @ dC_di @ inv_C @ dC_dj
                #matrix2 = dmu_di.T @ inv_C @ dmu_dj
                
                #F_cov_only[i, j] = 0.5 * np.trace(matrix1)
                #F_mu[i, j]  = matrix2.item()
                #F[i, j] = F_cov_only[i,j] + F_mu[i, j]

                val = all_derivs[a] @ inv_C @ all_derivs[b]
                F[a, b] = F[b, a] = val

        full_cov = np.linalg.inv(F)
        geo_cov = full_cov[:n_geo, :n_geo]

        if print_summary:
            n_gl = len(self._gl_flat)
            print(f"\nLR Fisher forecast ({n_amp} free amplitudes, "
                  f"{n_gl} GL constraints, marginalized)")
            for i, name in enumerate(self.geo_params):
                print(f"  sigma({name}) = {np.sqrt(geo_cov[i, i]):.4e}")

        self.F_lr = F
        self.cov_lr = full_cov
        self.geo_cov_lr = geo_cov
        return F, geo_cov

# ---------------------------------------------------------------------------------------------------------------------------------------#

        # ---------------------------------------------------------------------
    # Helper: which flat-vector indices belong to the CG/GL sub-vector
    # ---------------------------------------------------------------------
    def _sub_indices(lr):
        cg_idx = [g['cg_idx'] for g in lr.cg_gl_map]
        gl_idx = [gl['gl_idx'] for gl in lr._gl_flat]   # <-- from _gl_flat, not cg_gl_map
        return np.array(cg_idx + gl_idx)
    
    def _sub_indices_and_cov(lr, C):
        idx = _sub_indices(lr)
        return idx, C[np.ix_(idx, idx)]
        
    # ---------------------------------------------------------------------
    # The profile-likelihood fit: theta_geo is the only thing ever sampled
    # or optimized over; the shared amplitudes A are solved for exactly
    # (GLS) and marginalized out analytically at every trial theta_geo.
    # ---------------------------------------------------------------------
    def profiled_chi2(theta_values, geo_params, lr, d_sub, C_sub_inv):
        """
        Evaluate the GLS-profiled chi^2 at one trial theta_geo point.
    
        Steps:
          1. Build the trial cosmology (fiducial values, with geo_params
             overridden by theta_values).
          2. Compute r(theta_geo) for every lens/source pairing via
             lr.compute_r_vector -- this is the ONLY place Pmm-independent
             geometry enters; nothing here ever touches the wrong (or right)
             Pmm used to build the mock data, only the frozen fiducial one
             baked into lr.compute_r_vector via lr._pk_frozen.
          3. Build the linear design matrix X: CG rows get a 1 in their own
             amplitude's column, GL rows get r in THEIR group's column (so
             multiple GL rows can share one amplitude column).
          4. Solve for the amplitudes A_hat that best fit d_sub given X,
             weighted by the real covariance (ordinary GLS: A_hat =
             (X^T C^-1 X)^-1 X^T C^-1 d).
          5. Return the resulting chi^2 of the residual -- this already has
             every amplitude integrated out, exactly (not approximately),
             because the model is linear in A and the likelihood is Gaussian.
        """
        trial = dict(lr.fiducial_dict)
        for name, val in zip(geo_params, theta_values):
            trial[name] = val
        cosmo_trial = lr._make_cosmo(trial)
    
        r = lr.compute_r_vector(cosmo_trial)
        n_amp = len(lr.cg_gl_map)
        n_gl = len(lr._gl_flat)
        n_sub = n_amp + n_gl
    
        X = np.zeros((n_sub, n_amp))
        for g_idx in range(n_amp):
            X[g_idx, g_idx] = 1.0                # CG rows: first n_amp rows
        for k, gl in enumerate(lr._gl_flat):
            X[n_amp + k, gl['group_idx']] = r[k]  # GL rows: rest, r in their group's column
    
        XtCinv = X.T @ C_sub_inv
        A_hat = np.linalg.solve(XtCinv @ X, XtCinv @ d_sub)
        resid = d_sub - X @ A_hat
        return float(resid @ C_sub_inv @ resid)
    
    
    def fit_theta_geo_profiled(lr, d_sub, C_sub, geo_params=None, x0=None):
        """
        Minimize profiled_chi2 over theta_geo only -- A never appears as an
        explicit optimization dimension. Nelder-Mead since profiled_chi2 isn't
        handed analytic gradients; fine for the 1-2 dimensional theta_geo this
        is meant for.
        """
        geo_params = geo_params or lr.geo_params
        C_sub_inv = np.linalg.inv(C_sub)
        x0 = x0 or [lr.fiducial_dict[p] for p in geo_params]
    
        result = minimize(profiled_chi2, x0=x0, args=(geo_params, lr, d_sub, C_sub_inv),
                           method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6})
        return dict(zip(geo_params, result.x)), result
    
    
    # ---------------------------------------------------------------------
    # Build a mock using a given (possibly wrong) matter power spectrum
    # ---------------------------------------------------------------------
    def build_mock_with_pmm(lr, pmm_cosmology, noiseless=True):
        """
        The FULL, physically-motivated theory vector at pmm_cosmology -- real
        Pmm, real bias, everything -- restricted down to just the CG/GL rows.
        This is meant to stand in for "what the sky actually looks like" if
        pmm_cosmology's matter power spectrum were the true one; noiseless=True
        means it's the noise-free model prediction, not a noisy draw.
        """
        mu_full = lr.build_theory_vector(pmm_cosmology, noiseless=noiseless)
        return mu_full[_sub_indices(lr)]
    
    
    # ---------------------------------------------------------------------
    # The actual immunity test
    # ---------------------------------------------------------------------
    def test_pmm_immunity(lr, wrong_pmm_cosmology, C=None, n_sigma_flag=0.3):
        """
        Core question: if the TRUE matter power spectrum were wrong_pmm_cosmology's
        (rather than lr.cosmology's fiducial one), would fitting theta_geo with
        the ratio method still recover the right answer?
    
        Procedure:
          1. Build (or reuse) the full covariance matrix, restricted to CG/GL.
          2. Build two mocks at the SAME true theta_geo: one using the real
             (fiducial) Pmm, one using the deliberately WRONG Pmm.
          3. Fit theta_geo against each mock via the profile-likelihood fit
             above -- note this fit never sees which Pmm generated the mock;
             it only ever uses the frozen fiducial Pmm inside lr.compute_r_vector
             to build r, and otherwise reads everything else straight from the
             mock data itself.
          4. Compare the two fitted theta_geo values. If the ratio method's
             immunity claim holds, they should agree to well within the
             Fisher-forecasted sigma on theta_geo -- a real, biased fit would
             instead show a shift many sigma large.
        """
        p = lr.survey_params
        if C is None:
            cov_obj, _, _ = build_covariance_from_data(lr.cosmology, lr.lens_data, lr.source_data, **p)
            C = cov_obj.matrix
        _, C_sub = _sub_indices_and_cov(lr, C)
    
        d_wrong = build_mock_with_pmm(lr, wrong_pmm_cosmology)
        best_wrong, _ = fit_theta_geo_profiled(lr, d_wrong, C_sub)
    
        d_fid = build_mock_with_pmm(lr, lr.cosmology)
        best_fid, _ = fit_theta_geo_profiled(lr, d_fid, C_sub)
    
        _, geo_cov = lr.make_lr_fisher_matrix(C=C)
        sigma = np.sqrt(np.diag(geo_cov))
    
        print("\n[Pmm immunity test]")
        all_pass = True
        for i, name in enumerate(lr.geo_params):
            shift = best_wrong[name] - best_fid[name]
            n_sig = abs(shift) / sigma[i]
            flag = n_sig > n_sigma_flag
            all_pass &= not flag
            print(f"  {name}: fiducial-mock fit = {best_fid[name]:.6f}, "
                  f"wrong-Pmm-mock fit = {best_wrong[name]:.6f}, "
                  f"shift = {shift:.2e} ({n_sig:.2f} sigma) {'<-- FLAG' if flag else ''}")
        print("  " + ("PASS -- recovered theta_geo is stable against the wrong Pmm"
                       if all_pass else
                       "FAIL -- theta_geo shifted more than expected; check kernel "
                       "narrowness and pk_override wiring."))
        return best_fid, best_wrong, sigma

    # ------------------------------------------------------------------
    # 6.5. Lean mu-only derivatives: same 5-point stencil as the parent
    #    class's get_derivatives, but skips the covariance stencil (4 extra
    #    build_covariance_from_data calls per parameter) entirely, since
    #    make_joint_fisher_matrix never uses C_derivatives. With
    #    cmb_primaries=True this roughly halves runtime -- each of those
    #    discarded covariance-derivative evaluations was triggering a full
    #    CAMB primary-spectrum run for nothing.
    # ------------------------------------------------------------------
    def get_mu_derivatives_only(self, desired_params):
        mu_derivatives = {}
        for param in desired_params:
            step = self.step_dict.get(param, 1e-3)
            p_up1 = self.fiducial_dict.copy(); p_up1[param] += step
            p_up2 = self.fiducial_dict.copy(); p_up2[param] += 2 * step
            p_dn1 = self.fiducial_dict.copy(); p_dn1[param] -= step
            p_dn2 = self.fiducial_dict.copy(); p_dn2[param] -= 2 * step

            mu_up1 = self.build_theory_vector(self._make_cosmo(p_up1), noiseless=True)
            mu_up2 = self.build_theory_vector(self._make_cosmo(p_up2), noiseless=True)
            mu_dn1 = self.build_theory_vector(self._make_cosmo(p_dn1), noiseless=True)
            mu_dn2 = self.build_theory_vector(self._make_cosmo(p_dn2), noiseless=True)

            mu_derivatives[param] = (-mu_up2 + 8.0 * mu_up1 - 8.0 * mu_dn1 + mu_dn2) / (12.0 * step)
        return mu_derivatives

    # ------------------------------------------------------------------
    # 7. Joint fit with CMB primaries (TT/EE/ET). Requires this forecaster
    #    to have been built with desired_spectra including 'CG','GL', and
    #    whichever of 'TT'/'EE'/'ET' you want, plus cmb_primaries=True.
    # ------------------------------------------------------------------
    def _primary_pairs_present(self):
        candidates = [('E', 'E'), ('T', 'T'), ('E', 'T')]
        return [pair for pair in candidates if pair in self.final_f_map.pair_to_index]

    def _primary_pair_indices(self):
        """Flat indices of every requested primary pair (TT/EE/ET), in the
        same flat-vector convention as everything else in this class."""
        pairs_used = self._primary_pairs_present()
        idx = []
        for pair in pairs_used:
            base = self.final_f_map.pair_to_index[pair] * self.num_binned_ells
            idx.extend(range(base, base + self.num_binned_ells))
        return np.array(idx, dtype=int), pairs_used

    def _bin_all_ells(self, unbinned_cls):
        """Bin an entire unbinned Cl array into num_binned_ells values, one
        call per ell bin to _bin_one_ell (same mode-weighted convention)."""
        return np.array([self._bin_one_ell(unbinned_cls, b) for b in range(self.num_binned_ells)])

    def _compute_primary_cls(self, cosmology):
        """
        TT, EE, ET/TE directly via CAMB -- NO CCL tracer building at all.
        Mirrors the cmb_primaries block inside build_spectra_dict, but
        skips everything that block's surrounding code does that we don't
        need here (building lens/source/CMB-lensing tracers, and the CCL
        angular_cl loop over every lensing/galaxy pair). Primaries don't
        depend on any of that, so computing it was always wasted work for
        this specific derivative.
        """
        cosmology.compute_growth()
        h = cosmology['h']
        ombh2 = cosmology['Omega_b'] * h ** 2
        omch2 = cosmology['Omega_c'] * h ** 2
        A_s = cosmology['A_s']
        n_s = cosmology['n_s']
        Omega_k = cosmology['Omega_k']
        w0 = cosmology['w0']
        wa = cosmology['wa']
        Neff = cosmology['Neff']
        m_nu = cosmology['m_nu']
        T_CMB = cosmology['T_CMB']
        l_max = np.max(self.ells)

        if hasattr(m_nu, '__len__') or isinstance(m_nu, (list, np.ndarray)):
            m_nu = np.sum(m_nu)

        pars = camb.CAMBparams()
        pars.set_cosmology(H0=h * 100, ombh2=ombh2, omch2=omch2, omk=Omega_k,
                            mnu=m_nu, nnu=Neff, TCMB=T_CMB)
        pars.InitPower.set_params(As=A_s, ns=n_s)
        pars.set_dark_energy(w=w0, wa=wa, dark_energy_model='ppf')
        pars.set_for_lmax(l_max, lens_potential_accuracy=0)

        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
        lensed_cls = powers['total']
        camb_l = np.arange(lensed_cls.shape[0])

        return {
            ('E', 'E'): np.interp(self.ells, camb_l, lensed_cls[:, 1]),
            ('T', 'T'): np.interp(self.ells, camb_l, lensed_cls[:, 0]),
            ('E', 'T'): np.interp(self.ells, camb_l, lensed_cls[:, 3]),
        }

    def get_primary_mu_derivatives_only(self, desired_params):
        """
        5-point stencil, but for ONLY the primary (TT/EE/ET) bandpowers,
        via _compute_primary_cls -- no CCL tracers, no lensing/galaxy pairs.
        Returns {param: array of shape (len(primary_idx),)}, already
        reduced to just the primary rows, in the same order
        _primary_pair_indices builds them in.
        """
        pairs_used = self._primary_pairs_present()
        mu_derivatives = {}
        for param in desired_params:
            step = self.step_dict.get(param, 1e-3)
            p_up1 = self.fiducial_dict.copy(); p_up1[param] += step
            p_up2 = self.fiducial_dict.copy(); p_up2[param] += 2 * step
            p_dn1 = self.fiducial_dict.copy(); p_dn1[param] -= step
            p_dn2 = self.fiducial_dict.copy(); p_dn2[param] -= 2 * step

            cls_up1 = self._compute_primary_cls(self._make_cosmo(p_up1))
            cls_up2 = self._compute_primary_cls(self._make_cosmo(p_up2))
            cls_dn1 = self._compute_primary_cls(self._make_cosmo(p_dn1))
            cls_dn2 = self._compute_primary_cls(self._make_cosmo(p_dn2))

            chunks = []
            for pair in pairs_used:
                b_up1 = self._bin_all_ells(cls_up1[pair])
                b_up2 = self._bin_all_ells(cls_up2[pair])
                b_dn1 = self._bin_all_ells(cls_dn1[pair])
                b_dn2 = self._bin_all_ells(cls_dn2[pair])
                chunks.append((-b_up2 + 8 * b_up1 - 8 * b_dn1 + b_dn2) / (12.0 * step))
            mu_derivatives[param] = np.concatenate(chunks)
        return mu_derivatives

    def make_joint_fisher_matrix(self, primary_params, C=None,
                                  finite_diff_step=None, print_summary=False):
        """
        Joint Fisher matrix combining:
          - the LR ratio model on CG/GL (theta_geo + shared amplitudes A_i)
          - the standard physically-motivated model on CMB primaries,
            sensitive to theta_geo AND primary_params (e.g. 'A_s','n_s',
            'h','Omega_b','Omega_c').
        """
        p = self.survey_params
        if C is None:
            cov_obj, _, _ = build_covariance_from_data(
                self.cosmology, self.lens_data, self.source_data, **p)
            C = cov_obj.matrix
        inv_C = np.linalg.inv(C)
        n_total = C.shape[0]

        primary_idx, primary_pairs_used = self._primary_pair_indices()
        if len(primary_idx) == 0:
            raise ValueError(
                "No primary pairs (TT/EE/ET) found in final_f_map.pairs -- "
                "check desired_spectra and cmb_primaries=True."
            )

        combined_params = list(self.geo_params) + [
            pp for pp in primary_params if pp not in self.geo_params
        ]

        # Primary-only derivatives via direct CAMB calls -- no CCL tracers,
        # no lensing/galaxy pair computation at all (see
        # get_primary_mu_derivatives_only). 
        primary_derivs = self.get_primary_mu_derivatives_only(combined_params)

        r_fid = self.compute_r_vector(self.cosmology)
        A_fid = self.compute_fiducial_amplitudes()
        X_fid = self._design_matrix(r_fid, n_total)
        n_amp = len(self.cg_gl_map)

        combined_derivs = {}
        for param in combined_params:
            vec = np.zeros(n_total)
            vec[primary_idx] = primary_derivs[param]  # primaries: always standard

            if param in self.geo_params:  # CG/GL: ONLY the LR mechanism
                h = finite_diff_step or self.step_dict.get(param, 1e-3)
                up = dict(self.fiducial_dict); up[param] = self.fiducial_dict[param] + h
                dn = dict(self.fiducial_dict); dn[param] = self.fiducial_dict[param] - h
                r_up = self.compute_r_vector(self._make_cosmo(up))
                r_dn = self.compute_r_vector(self._make_cosmo(dn))
                dr = (r_up - r_dn) / (2 * h)
                for k, gl in enumerate(self._gl_flat):
                    vec[gl['gl_idx']] = dr[k] * A_fid[gl['group_idx']]
            # CG rows are left at 0 for every parameter -- amplitude-only, always.

            combined_derivs[param] = vec

        all_derivs = [combined_derivs[pname] for pname in combined_params]
        all_derivs += [X_fid[:, k] for k in range(n_amp)]

        n_nonamp = len(combined_params)
        n_par = n_nonamp + n_amp
        F = np.zeros((n_par, n_par))
        for a in range(n_par):
            for b in range(a, n_par):
                val = all_derivs[a] @ inv_C @ all_derivs[b]
                F[a, b] = F[b, a] = val

        full_cov = np.linalg.inv(F)
        param_cov = full_cov[:n_nonamp, :n_nonamp]

        if print_summary:
            print(f"\nJoint LR + primaries Fisher forecast "
                  f"({n_amp} amplitudes marginalized, primaries used: {primary_pairs_used})")
            for i, name in enumerate(combined_params):
                print(f"  sigma({name}) = {np.sqrt(param_cov[i, i]):.4e}")

        self.F_joint = F
        self.cov_joint = full_cov
        self.combined_params = combined_params
        return F, param_cov, combined_params
        
