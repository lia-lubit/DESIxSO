# testing the time-taking components of PyCCL

# import tools
from functions_and_classes import calculate_Cls_w_emulator
from functions_and_classes import make_Pk2D
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
import numpy as np

# standard library
import os
import sys
import inspect
import yaml
import time

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

lin_emu = CPJ(probe='mpk_lin')
boost_emu = CPJ(probe='mpk_boost')

DESI_lens_data = np.loadtxt("data/des_nz_gamma.txt")
HSC_dist_0_3 = np.load("data/distributions_0.3_3_vF.npz")

Om0 = 0.3027   # matter density
H0 = 68.17     # Hubble [km/s/Mpc]
h = 0.6817
Ode0 = 1 - Om0 # dark energy density
A_s = 2.1e-9 # Using A_s, consistent with emulator input
Omega_k = 0
Omega_b = 0.045
n_s=0.96
Omega_c = 0.27

flat_LCDM_cosmology2_timed = ccl.Cosmology(
    Omega_k=Omega_k, Omega_c=Omega_c, Omega_b=Omega_b, h=h, A_s=A_s, # Use A_s here
    n_s=n_s, transfer_function='boltzmann_camb'
)

z_grid = np.linspace(0, 10, 500)
Pk2D_full_timed = make_Pk2D(
    flat_LCDM_cosmology2_timed, # Use the same cosmology to ensure parameter consistency
    linear_emulator=lin_emu,
    boost_emulator=boost_emu,
    z_arr=z_grid,
    cmin=3.13,
    eta_0=0.68
)

emulator_Cl_LL = calculate_Cls_w_emulator(
    cosmology=flat_LCDM_cosmology2_timed,
    lens_data=DESI_lens_data,
    source_data=HSC_dist_0_3,
    correlation_types=['LL'],
    Pk2D_object = Pk2D_full_timed
)
