#!/usr/bin/env python
"""
Comprehensive dust model validation test.

Creates mock photometry using NATIVE FSPS dust emission, then fits with:
1. Native prospector (SSPBasis with add_dust_emission=True) - should match exactly
2. CigaleDustSSPBasis with fsps_dl2007 model - should match exactly  
3. CigaleDustSSPBasis with other models (dl2007, dl2014, themis) - will differ

This validates that:
- fsps_dl2007 exactly reproduces native FSPS behavior
- Other CIGALE models work correctly (different but reasonable fits)
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

import warnings
warnings.filterwarnings('ignore')

import fsps
import sedpy.observate
from sedpy.observate import Filter
import shutil

# Get sedpy filter directory for installing custom filters
SEDPY_FILTER_DIR = Path(os.path.dirname(sedpy.observate.__file__)) / 'data' / 'filters'

from prospect.observation import Photometry, IntrinsicSpectrum
from prospect.models.templates import TemplateLibrary
from prospect.models.sedmodel import SpecModel
from prospect.fitting import lnprobfn
from prospect.sources import SSPBasis, FastStepBasis
from prospect.sources.cigale_duste_basis import CigaleDustSSPBasis
from scipy.optimize import dual_annealing

C_AA = 2.998e18


def optimize_model(observations, model, sps, maxiter=500):
    """
    Optimize model parameters using dual_annealing with prospector's lnprobfn.
    
    Parameters
    ----------
    observations : list of Observation objects
        The observations to fit (Photometry, etc.)
    model : SpecModel
        The prospector model
    sps : SSPBasis or similar
        The stellar population synthesis object
    maxiter : int
        Maximum iterations for dual_annealing
    
    Returns best-fit theta and -lnprob (which is chi2/2 + constants for Gaussian likelihood).
    """
    # Get bounds from priors
    bounds = []
    for p in model.free_params:
        prior = model.config_dict[p].get('prior')
        if hasattr(prior, 'params') and 'mini' in prior.params and 'maxi' in prior.params:
            bounds.append((prior.params['mini'], prior.params['maxi']))
        else:
            bounds.append((0.1, 10.0))
    
    # Use prospector's lnprobfn with negative=True for minimization
    def objective(theta):
        return lnprobfn(theta, model=model, observations=observations, 
                        sps=sps, negative=True)
    
    # Run dual_annealing
    result = dual_annealing(objective, bounds, maxiter=maxiter, seed=42)
    
    return result.x, result.fun


def get_sed_prediction(model, theta, sps, wave_grid_aa):
    """
    Get full SED prediction using IntrinsicSpectrum.
    
    Parameters
    ----------
    model : SpecModel
        The prospector model
    theta : array
        Parameter vector
    sps : SSPBasis or similar
        The SPS object
    wave_grid_aa : array
        Wavelength grid in Angstroms
    
    Returns
    -------
    maggies : array
        Model SED in maggies at the wavelength grid
    """
    full_sed_obs = IntrinsicSpectrum(
        wavelength=wave_grid_aa,
        flux=np.ones_like(wave_grid_aa),
        uncertainty=np.ones_like(wave_grid_aa),
        name='full_sed'
    )
    full_sed_obs.redshift = 0.0
    full_sed_obs.rectify()
    
    predictions, _ = model.predict(theta, [full_sed_obs], sps=sps)
    return predictions[0]


# =============================================================================
# Custom filter handling (for ALMA bands)
# =============================================================================
CUSTOM_FILTER_SPECS = {
    "alma_band_6": (1090.4, 1247.8, 1420.7),   # 211-275 GHz
    "alma_band_7": (803.8, 897.1, 1090.4),     # 275-373 GHz
}


def write_custom_filter_file(filter_name, wave_lo_micron, wave_hi_micron, output_dir, n_points=100):
    """Write a custom top-hat filter file to disk in sedpy .par format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wave_lo_AA = wave_lo_micron * 1e4
    wave_hi_AA = wave_hi_micron * 1e4
    pad = 0.01 * (wave_hi_AA - wave_lo_AA)
    wavelength_AA = np.concatenate([
        [wave_lo_AA - pad],
        np.linspace(wave_lo_AA, wave_hi_AA, n_points),
        [wave_hi_AA + pad]
    ])
    transmission = np.concatenate([[0.0], np.ones(n_points), [0.0]])
    filter_path = output_dir / f"{filter_name}.par"
    with open(filter_path, 'w') as f:
        f.write(f"# Custom top-hat filter: {filter_name}\n")
        for w, t in zip(wavelength_AA, transmission):
            f.write(f"{w:.2f}  {t:.6f}\n")
    return filter_path


def ensure_custom_filters(filter_names, output_dir):
    """Ensure custom filter files exist for ALMA bands."""
    for filter_name in filter_names:
        if filter_name in CUSTOM_FILTER_SPECS:
            wave_lo, wave_cen, wave_hi = CUSTOM_FILTER_SPECS[filter_name]
            local_path = write_custom_filter_file(filter_name, wave_lo, wave_hi, output_dir)
            sedpy_path = SEDPY_FILTER_DIR / f"{filter_name}.par"
            if not sedpy_path.exists():
                shutil.copy(local_path, sedpy_path)
                print(f"  Installed custom filter: {filter_name}")


def get_dust_emission_filters():
    """Return comprehensive filter set for dust emission fitting."""
    return [
        # Optical/NIR for stellar continuum
        'sdss_r0', 'sdss_i0', 'sdss_z0',
        'twomass_J', 'twomass_H', 'twomass_Ks',
        # WISE
        'wise_w1', 'wise_w2', 'wise_w3', 'wise_w4',
        # Spitzer IRAC
        'spitzer_irac_ch1', 'spitzer_irac_ch2', 'spitzer_irac_ch3', 'spitzer_irac_ch4',
        # Spitzer MIPS
        'spitzer_mips_24', 'spitzer_mips_70', 'spitzer_mips_160',
        # Herschel PACS
        'herschel_pacs_70', 'herschel_pacs_100', 'herschel_pacs_160',
        # Herschel SPIRE
        'herschel_spire_250', 'herschel_spire_350', 'herschel_spire_500',
        # ALMA
        'alma_band_7', 'alma_band_6',
    ]


# =============================================================================
# Model building helpers
# =============================================================================
def build_native_fsps_model(free_dust_params=True):
    """Build model using native FSPS dust emission."""
    from prospect.models.priors import TopHat
    
    model_params = TemplateLibrary["parametric_sfh"]
    
    # Add native FSPS dust emission params
    model_params['add_dust_emission'] = {'N': 1, 'init': True, 'isfree': False}
    model_params['duste_qpah'] = {'N': 1, 'init': 3.5, 'isfree': free_dust_params,
                                   'prior': TopHat(mini=0.5, maxi=7.0)}
    model_params['duste_umin'] = {'N': 1, 'init': 1.0, 'isfree': free_dust_params,
                                   'prior': TopHat(mini=0.1, maxi=25.0)}
    model_params['duste_gamma'] = {'N': 1, 'init': 0.01, 'isfree': free_dust_params,
                                    'prior': TopHat(mini=0.0, maxi=1.0)}
    
    # Fix SFH parameters
    model_params['tau'] = {'N': 1, 'init': 1.0, 'isfree': False}
    model_params['tage'] = {'N': 1, 'init': 5.0, 'isfree': False}
    model_params['mass'] = {'N': 1, 'init': 1e10, 'isfree': False}
    model_params['logzsol'] = {'N': 1, 'init': 0.0, 'isfree': False}
    model_params['dust2'] = {'N': 1, 'init': 0.3, 'isfree': False}
    model_params['zred'] = {'N': 1, 'init': 0.0, 'isfree': False}
    
    return SpecModel(model_params)


def build_cigale_model(dust_model, free_dust_params=True):
    """Build model using CigaleDustSSPBasis with specified dust model."""
    from prospect.models.priors import TopHat
    
    model_params = TemplateLibrary["parametric_sfh"]
    
    # Remove native dust emission
    model_params['add_dust_emission'] = {'N': 1, 'init': False, 'isfree': False}
    
    # Add CIGALE dust emission params based on model
    prefix = 'duste'  # Use same prefix as native FSPS for consistency
    
    if dust_model in ['fsps_dl2007', 'dl2007']:
        model_params[f'{prefix}_qpah'] = {'N': 1, 'init': 3.5, 'isfree': free_dust_params,
                                          'prior': TopHat(mini=0.5, maxi=7.0)}
        model_params[f'{prefix}_umin'] = {'N': 1, 'init': 1.0, 'isfree': free_dust_params,
                                          'prior': TopHat(mini=0.1, maxi=25.0)}
        model_params[f'{prefix}_gamma'] = {'N': 1, 'init': 0.01, 'isfree': free_dust_params,
                                           'prior': TopHat(mini=0.0, maxi=1.0)}
        if dust_model == 'dl2007':
            model_params[f'{prefix}_umax'] = {'N': 1, 'init': 1e6, 'isfree': False}
            
    elif dust_model == 'dl2014':
        model_params[f'{prefix}_qpah'] = {'N': 1, 'init': 3.5, 'isfree': free_dust_params,
                                          'prior': TopHat(mini=0.5, maxi=7.0)}
        model_params[f'{prefix}_umin'] = {'N': 1, 'init': 1.0, 'isfree': free_dust_params,
                                          'prior': TopHat(mini=0.1, maxi=25.0)}
        model_params[f'{prefix}_alpha'] = {'N': 1, 'init': 2.0, 'isfree': False}
        model_params[f'{prefix}_gamma'] = {'N': 1, 'init': 0.01, 'isfree': free_dust_params,
                                           'prior': TopHat(mini=0.0, maxi=1.0)}
        
    elif dust_model == 'themis':
        model_params[f'{prefix}_qhac'] = {'N': 1, 'init': 0.17, 'isfree': free_dust_params,
                                          'prior': TopHat(mini=0.02, maxi=0.25)}
        model_params[f'{prefix}_umin'] = {'N': 1, 'init': 1.0, 'isfree': free_dust_params,
                                          'prior': TopHat(mini=0.1, maxi=25.0)}
        model_params[f'{prefix}_alpha'] = {'N': 1, 'init': 2.0, 'isfree': False}
        model_params[f'{prefix}_gamma'] = {'N': 1, 'init': 0.01, 'isfree': free_dust_params,
                                           'prior': TopHat(mini=0.0, maxi=1.0)}
    
    # Fix SFH parameters
    model_params['tau'] = {'N': 1, 'init': 1.0, 'isfree': False}
    model_params['tage'] = {'N': 1, 'init': 5.0, 'isfree': False}
    model_params['mass'] = {'N': 1, 'init': 1e10, 'isfree': False}
    model_params['logzsol'] = {'N': 1, 'init': 0.0, 'isfree': False}
    model_params['dust2'] = {'N': 1, 'init': 0.3, 'isfree': False}
    model_params['zred'] = {'N': 1, 'init': 0.0, 'isfree': False}
    
    return SpecModel(model_params)


# =============================================================================
# Main test
# =============================================================================
def run_validation_test():
    """Run comprehensive dust model validation test."""
    print("=" * 70)
    print("DUST MODEL VALIDATION TEST")
    print("Mock data: Native FSPS dust emission")
    print("Fitting with: Native FSPS, fsps_dl2007, dl2007, dl2014, themis")
    print("=" * 70)
    print()
    
    # Setup
    custom_filter_dir = Path(script_dir) / 'custom_filters'
    custom_filter_dir.mkdir(exist_ok=True)
    filternames = get_dust_emission_filters()
    ensure_custom_filters(filternames, custom_filter_dir)
    filters = [Filter(f) for f in filternames]
    filter_waves = np.array([f.wave_effective for f in filters]) / 1e4  # microns
    
    print(f"Using {len(filters)} filters from {filter_waves.min():.1f} to {filter_waves.max():.0f} μm")
    print()
    
    # =========================================================================
    # Generate mock data with NATIVE FSPS
    # =========================================================================
    print("-" * 50)
    print("Generating mock data with NATIVE FSPS dust emission...")
    
    true_params = {
        'duste_qpah': 3.5,
        'duste_umin': 1.0,
        'duste_gamma': 0.01,
    }
    print(f"True params: qpah={true_params['duste_qpah']}, umin={true_params['duste_umin']}, gamma={true_params['duste_gamma']}")
    
    # Create native FSPS SPS
    sps_mock = SSPBasis(zcontinuous=1)
    sps_mock.ssp.params['add_dust_emission'] = True
    
    # Build mock model with TRUE parameters (not free)
    model_mock = build_native_fsps_model(free_dust_params=False)
    model_mock.params['duste_qpah'] = true_params['duste_qpah']
    model_mock.params['duste_umin'] = true_params['duste_umin']
    model_mock.params['duste_gamma'] = true_params['duste_gamma']
    
    # Generate mock photometry using prospector's built-in methods
    theta_mock = model_mock.theta.copy()
    
    # Use model.predict_phot which handles all conversions correctly
    model_mock.predict_init(theta_mock, sps=sps_mock)
    phot_true = model_mock.predict_phot(filters)
    
    # Add noise
    noise_level = 0.05  # 5% errors
    phot_unc = np.abs(phot_true) * noise_level
    np.random.seed(42)
    phot_noisy = phot_true + np.random.normal(0, phot_unc)
    phot_noisy = np.maximum(phot_noisy, phot_true * 0.01)  # Keep positive
    
    obs = Photometry(filters=filters, flux=phot_noisy, uncertainty=phot_unc)
    print(f"Generated {len(filters)} photometric points with {noise_level*100:.0f}% errors")
    print(f"Flux range: {phot_true.min():.2e} to {phot_true.max():.2e} maggies")
    print()
    
    # =========================================================================
    # Fit with each model
    # =========================================================================
    results = {}
    
    # Wavelength grid for SED plotting (0.3 - 500 microns in Angstroms)
    wave_grid_aa = np.logspace(np.log10(3000), np.log10(5e6), 500)
    
    # --- 1. Native FSPS ---
    print("-" * 50)
    print("1. Fitting with NATIVE FSPS...")
    
    model_native = build_native_fsps_model(free_dust_params=True)
    sps_native_fit = SSPBasis(zcontinuous=1)
    sps_native_fit.ssp.params['add_dust_emission'] = True
    
    # Use dual_annealing optimizer with lnprobfn
    best_theta, neg_lnprob = optimize_model([obs], model_native, sps_native_fit, maxiter=50)
    labels = model_native.theta_labels()
    
    # Get SED and photometry predictions
    sed_native = get_sed_prediction(model_native, best_theta, sps_native_fit, wave_grid_aa)
    model_native.predict_init(best_theta, sps=sps_native_fit)
    phot_native = model_native.predict_phot(filters)
    
    results['Native FSPS'] = {
        'qpah': best_theta[labels.index('duste_qpah')],
        'umin': best_theta[labels.index('duste_umin')],
        'gamma': best_theta[labels.index('duste_gamma')],
        'model': model_native,
        'sps': sps_native_fit,
        'theta': best_theta,
        'color': 'black',
        'neg_lnprob': neg_lnprob,
        'sed': sed_native,
        'phot': phot_native,
    }
    print(f"   qpah={results['Native FSPS']['qpah']:.3f}, umin={results['Native FSPS']['umin']:.3f}, gamma={results['Native FSPS']['gamma']:.4f} (-lnP={neg_lnprob:.1f})")
    
    # --- 2. FSPS DL2007 (should match native) ---
    print("-" * 50)
    print("2. Fitting with CigaleDustSSPBasis (fsps_dl2007)...")
    
    model_fsps_dl = build_cigale_model('fsps_dl2007', free_dust_params=True)
    sps_fsps_dl = CigaleDustSSPBasis(dust_emission_model='fsps_dl2007', zcontinuous=1)
    
    best_theta, neg_lnprob = optimize_model([obs], model_fsps_dl, sps_fsps_dl, maxiter=50)
    labels = model_fsps_dl.theta_labels()
    
    sed_fsps = get_sed_prediction(model_fsps_dl, best_theta, sps_fsps_dl, wave_grid_aa)
    model_fsps_dl.predict_init(best_theta, sps=sps_fsps_dl)
    phot_fsps = model_fsps_dl.predict_phot(filters)
    
    results['fsps_dl2007'] = {
        'qpah': best_theta[labels.index('duste_qpah')],
        'umin': best_theta[labels.index('duste_umin')],
        'gamma': best_theta[labels.index('duste_gamma')],
        'model': model_fsps_dl,
        'sps': sps_fsps_dl,
        'theta': best_theta,
        'color': 'blue',
        'neg_lnprob': neg_lnprob,
        'sed': sed_fsps,
        'phot': phot_fsps,
    }
    print(f"   qpah={results['fsps_dl2007']['qpah']:.3f}, umin={results['fsps_dl2007']['umin']:.3f}, gamma={results['fsps_dl2007']['gamma']:.4f} (-lnP={neg_lnprob:.1f})")
    
    # --- 3. CIGALE DL2007 ---
    print("-" * 50)
    print("3. Fitting with CigaleDustSSPBasis (dl2007)...")
    
    model_dl2007 = build_cigale_model('dl2007', free_dust_params=True)
    sps_dl2007 = CigaleDustSSPBasis(dust_emission_model='dl2007', zcontinuous=1)
    
    best_theta, neg_lnprob = optimize_model([obs], model_dl2007, sps_dl2007, maxiter=50)
    labels = model_dl2007.theta_labels()
    
    sed_dl2007 = get_sed_prediction(model_dl2007, best_theta, sps_dl2007, wave_grid_aa)
    model_dl2007.predict_init(best_theta, sps=sps_dl2007)
    phot_dl2007 = model_dl2007.predict_phot(filters)
    
    results['CIGALE DL2007'] = {
        'qpah': best_theta[labels.index('duste_qpah')],
        'umin': best_theta[labels.index('duste_umin')],
        'gamma': best_theta[labels.index('duste_gamma')],
        'model': model_dl2007,
        'sps': sps_dl2007,
        'theta': best_theta,
        'color': 'green',
        'neg_lnprob': neg_lnprob,
        'sed': sed_dl2007,
        'phot': phot_dl2007,
    }
    print(f"   qpah={results['CIGALE DL2007']['qpah']:.3f}, umin={results['CIGALE DL2007']['umin']:.3f}, gamma={results['CIGALE DL2007']['gamma']:.4f} (-lnP={neg_lnprob:.1f})")
    
    # --- 4. DL2014 ---
    print("-" * 50)
    print("4. Fitting with CigaleDustSSPBasis (dl2014)...")
    
    model_dl2014 = build_cigale_model('dl2014', free_dust_params=True)
    sps_dl2014 = CigaleDustSSPBasis(dust_emission_model='dl2014', zcontinuous=1)
    
    best_theta, neg_lnprob = optimize_model([obs], model_dl2014, sps_dl2014, maxiter=50)
    labels = model_dl2014.theta_labels()
    
    sed_dl2014 = get_sed_prediction(model_dl2014, best_theta, sps_dl2014, wave_grid_aa)
    model_dl2014.predict_init(best_theta, sps=sps_dl2014)
    phot_dl2014 = model_dl2014.predict_phot(filters)
    
    results['DL2014'] = {
        'qpah': best_theta[labels.index('duste_qpah')],
        'umin': best_theta[labels.index('duste_umin')],
        'gamma': best_theta[labels.index('duste_gamma')],
        'model': model_dl2014,
        'sps': sps_dl2014,
        'theta': best_theta,
        'color': 'orange',
        'neg_lnprob': neg_lnprob,
        'sed': sed_dl2014,
        'phot': phot_dl2014,
    }
    print(f"   qpah={results['DL2014']['qpah']:.3f}, umin={results['DL2014']['umin']:.3f}, gamma={results['DL2014']['gamma']:.4f} (-lnP={neg_lnprob:.1f})")
    
    # --- 5. THEMIS ---
    print("-" * 50)
    print("5. Fitting with CigaleDustSSPBasis (themis)...")
    
    model_themis = build_cigale_model('themis', free_dust_params=True)
    sps_themis = CigaleDustSSPBasis(dust_emission_model='themis', zcontinuous=1)
    
    best_theta, neg_lnprob = optimize_model([obs], model_themis, sps_themis, maxiter=50)
    labels = model_themis.theta_labels()
    
    sed_themis = get_sed_prediction(model_themis, best_theta, sps_themis, wave_grid_aa)
    model_themis.predict_init(best_theta, sps=sps_themis)
    phot_themis = model_themis.predict_phot(filters)
    
    results['THEMIS'] = {
        'qhac': best_theta[labels.index('duste_qhac')],
        'umin': best_theta[labels.index('duste_umin')],
        'gamma': best_theta[labels.index('duste_gamma')],
        'model': model_themis,
        'sps': sps_themis,
        'theta': best_theta,
        'color': 'red',
        'neg_lnprob': neg_lnprob,
        'sed': sed_themis,
        'phot': phot_themis,
    }
    print(f"   qhac={results['THEMIS']['qhac']:.3f}, umin={results['THEMIS']['umin']:.3f}, gamma={results['THEMIS']['gamma']:.4f} (-lnP={neg_lnprob:.1f})")
    
    # =========================================================================
    # Create visualization using IntrinsicSpectrum SEDs
    # =========================================================================
    print()
    print("-" * 50)
    print("Creating visualization...")
    
    wave_um = wave_grid_aa / 1e4  # microns
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Dust Model Validation: Mock Data from Native FSPS', fontsize=14, fontweight='bold')
    
    # --- Panel 1: Full SED comparison ---
    ax = axes[0, 0]
    for name, res in results.items():
        sed = res['sed']
        # Plot λFλ (maggies × Angstroms) vs wavelength
        ax.loglog(wave_um, sed * wave_grid_aa, '-', color=res['color'], 
                  lw=2.5 if 'Native' in name or 'fsps_dl' in name else 1.5, 
                  alpha=1.0 if 'Native' in name or 'fsps_dl' in name else 0.7,
                  label=f"{name} (-lnP={res['neg_lnprob']:.1f})")
    
    ax.errorbar(filter_waves, phot_noisy * filter_waves * 1e4, 
                yerr=phot_unc * filter_waves * 1e4,
                fmt='ko', ms=6, capsize=3, label='Mock data', zorder=10)
    
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel(r'$\lambda F_\lambda$ (maggies $\times$ Å)')
    ax.set_title('Full SED Comparison')
    ax.set_xlim(0.3, 2000)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # --- Panel 2: IR zoom ---
    ax = axes[0, 1]
    ir_mask = wave_um > 3
    for name, res in results.items():
        sed = res['sed']
        ax.loglog(wave_um[ir_mask], sed[ir_mask] * wave_grid_aa[ir_mask], '-', 
                  color=res['color'], 
                  lw=2.5 if 'Native' in name or 'fsps_dl' in name else 1.5, 
                  alpha=1.0 if 'Native' in name or 'fsps_dl' in name else 0.7,
                  label=name)
    
    ir_filter_mask = filter_waves > 3
    ax.errorbar(filter_waves[ir_filter_mask], 
                phot_noisy[ir_filter_mask] * filter_waves[ir_filter_mask] * 1e4, 
                yerr=phot_unc[ir_filter_mask] * filter_waves[ir_filter_mask] * 1e4,
                fmt='ko', ms=8, capsize=3, label='Mock data', zorder=10)
    
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel(r'$\lambda F_\lambda$ (maggies $\times$ Å)')
    ax.set_title('IR Region (Dust Emission)')
    ax.set_xlim(3, 500)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # --- Panel 3: Residuals ---
    ax = axes[1, 0]
    for name, res in results.items():
        residual = (res['phot'] - phot_noisy) / phot_unc
        ax.plot(filter_waves, residual, 'o-', color=res['color'], 
                ms=6, alpha=0.8, label=name)
    
    ax.axhline(0, color='k', ls='--', alpha=0.5)
    ax.axhline(-2, color='gray', ls=':', alpha=0.5)
    ax.axhline(2, color='gray', ls=':', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('Residual (σ)')
    ax.set_title('Fit Residuals')
    ax.set_xlim(0.3, 2000)
    ax.set_ylim(-5, 5)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # --- Panel 4: Parameter recovery table ---
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create table data
    table_data = [
        ['Model', 'qpah/qhac', 'umin', 'gamma', '-lnP'],
    ]
    
    for name, res in results.items():
        q_param = res.get('qpah', res.get('qhac', np.nan))
        table_data.append([
            name,
            f"{q_param:.2f}",
            f"{res['umin']:.2f}",
            f"{res['gamma']:.3f}",
            f"{res['neg_lnprob']:.1f}"
        ])
    
    # Add true values row
    table_data.append([
        'TRUE VALUES',
        f"{true_params['duste_qpah']:.2f}",
        f"{true_params['duste_umin']:.2f}",
        f"{true_params['duste_gamma']:.3f}",
        '---'
    ])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Color header row
    for j in range(5):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Color true values row
    for j in range(5):
        table[(len(table_data)-1, j)].set_facecolor('#E2EFDA')
        table[(len(table_data)-1, j)].set_text_props(fontweight='bold')
    
    # Highlight matching results (Native FSPS and fsps_dl2007)
    for i, name in enumerate(['Native FSPS', 'fsps_dl2007'], start=1):
        for j in range(5):
            table[(i, j)].set_facecolor('#DDEBF7')
    
    ax.set_title('Parameter Recovery Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plot_path = os.path.join(script_dir, 'dust_model_validation.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    
    # =========================================================================
    # Print summary
    # =========================================================================
    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print()
    print(f"True values: qpah={true_params['duste_qpah']}, umin={true_params['duste_umin']}, gamma={true_params['duste_gamma']}")
    print()
    print(f"{'Model':<18} {'q param':<10} {'umin':<10} {'gamma':<10} {'-lnP':<10}")
    print("-" * 58)
    
    for name, res in results.items():
        q_param = res.get('qpah', res.get('qhac', np.nan))
        print(f"{name:<18} {q_param:<10.3f} {res['umin']:<10.3f} {res['gamma']:<10.4f} {res['neg_lnprob']:<10.1f}")
    
    print()
    print("Expected behavior:")
    print("  - Native FSPS and fsps_dl2007 should give IDENTICAL results")
    print("  - Other models will give different but reasonable fits")
    print()
    
    # Check if fsps_dl2007 matches native
    native = results['Native FSPS']
    fsps_dl = results['fsps_dl2007']
    
    qpah_match = abs(native['qpah'] - fsps_dl['qpah']) < 0.1
    umin_match = abs(native['umin'] - fsps_dl['umin']) < 0.1
    gamma_match = abs(native['gamma'] - fsps_dl['gamma']) < 0.01
    
    if qpah_match and umin_match and gamma_match:
        print("✓ VALIDATION PASSED: fsps_dl2007 matches Native FSPS!")
    else:
        print("✗ VALIDATION FAILED: fsps_dl2007 differs from Native FSPS")
        print(f"  qpah diff: {abs(native['qpah'] - fsps_dl['qpah']):.3f}")
        print(f"  umin diff: {abs(native['umin'] - fsps_dl['umin']):.3f}")
        print(f"  gamma diff: {abs(native['gamma'] - fsps_dl['gamma']):.4f}")
    
    plt.show()
    return results


if __name__ == '__main__':
    run_validation_test()
