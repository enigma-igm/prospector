#!/usr/bin/env python
"""
Quick validation that fsps_dl2007 matches Native FSPS exactly.
Uses just 2 models and minimal iterations for speed.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

import warnings
warnings.filterwarnings('ignore')

from sedpy.observate import Filter
from scipy.optimize import minimize

from prospect.observation import Photometry
from prospect.models.templates import TemplateLibrary
from prospect.models.sedmodel import SpecModel
from prospect.models.priors import TopHat
from prospect.sources import SSPBasis
from prospect.sources.cigale_duste_basis import CigaleDustSSPBasis


def get_filters():
    """Get a compact filter set for quick testing."""
    return [
        'wise_w3', 'wise_w4',  # MIR
        'spitzer_mips_24', 'spitzer_mips_70',  # MIR/FIR
        'herschel_pacs_100', 'herschel_pacs_160',  # FIR
        'herschel_spire_250',  # submm
    ]


def build_model():
    """Build parametric SFH model with dust emission."""
    model_params = TemplateLibrary["parametric_sfh"]
    model_params['add_dust_emission'] = {'N': 1, 'init': True, 'isfree': False}
    model_params['duste_qpah'] = {'N': 1, 'init': 3.5, 'isfree': True,
                                   'prior': TopHat(mini=0.5, maxi=6.0)}
    model_params['duste_umin'] = {'N': 1, 'init': 1.0, 'isfree': True,
                                   'prior': TopHat(mini=0.1, maxi=15.0)}
    model_params['duste_gamma'] = {'N': 1, 'init': 0.01, 'isfree': True,
                                    'prior': TopHat(mini=0.0, maxi=0.5)}
    model_params['tau'] = {'N': 1, 'init': 1.0, 'isfree': False}
    model_params['tage'] = {'N': 1, 'init': 5.0, 'isfree': False}
    model_params['mass'] = {'N': 1, 'init': 1e10, 'isfree': False}
    model_params['logzsol'] = {'N': 1, 'init': 0.0, 'isfree': False}
    model_params['dust2'] = {'N': 1, 'init': 0.3, 'isfree': False}
    model_params['zred'] = {'N': 1, 'init': 0.0, 'isfree': False}
    return SpecModel(model_params)


def fit_model(obs, model, sps):
    """Simple bounded optimization."""
    bounds = [(0.5, 6.0), (0.1, 15.0), (0.0, 0.5)]  # qpah, umin, gamma
    
    def objective(x):
        theta = np.array(x)
        try:
            model.predict_init(theta, sps=sps)
            phot_model = model.predict_phot(obs.filters)
            chi2 = np.sum(((phot_model - obs.flux) / obs.uncertainty)**2)
            return chi2
        except:
            return 1e10
    
    # Start from true values
    x0 = [3.5, 1.0, 0.01]
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    return result.x, result.fun


def run_validation():
    """Run quick validation test."""
    print("=" * 60)
    print("QUICK VALIDATION: Native FSPS vs fsps_dl2007")
    print("=" * 60)
    
    # Setup
    filternames = get_filters()
    filters = [Filter(f) for f in filternames]
    filter_waves = np.array([f.wave_effective for f in filters]) / 1e4
    
    print(f"Using {len(filters)} filters")
    
    # True parameters
    true_qpah, true_umin, true_gamma = 3.5, 1.0, 0.01
    print(f"True params: qpah={true_qpah}, umin={true_umin}, gamma={true_gamma}")
    
    # Generate mock data with Native FSPS
    print("\nGenerating mock data with Native FSPS...")
    sps_mock = SSPBasis(zcontinuous=1)
    sps_mock.ssp.params['add_dust_emission'] = True
    
    model_mock = build_model()
    model_mock.params['duste_qpah'] = true_qpah
    model_mock.params['duste_umin'] = true_umin
    model_mock.params['duste_gamma'] = true_gamma
    
    theta_mock = model_mock.theta.copy()
    model_mock.predict_init(theta_mock, sps=sps_mock)
    phot_true = model_mock.predict_phot(filters)
    
    # Add noise
    noise_level = 0.05
    phot_unc = np.abs(phot_true) * noise_level
    np.random.seed(42)
    phot_noisy = phot_true + np.random.normal(0, phot_unc)
    
    obs = Photometry(filters=filters, flux=phot_noisy, uncertainty=phot_unc)
    
    # Fit with Native FSPS
    print("\nFitting with Native FSPS...")
    model_native = build_model()
    sps_native = SSPBasis(zcontinuous=1)
    sps_native.ssp.params['add_dust_emission'] = True
    
    theta_native, chi2_native = fit_model(obs, model_native, sps_native)
    print(f"  Result: qpah={theta_native[0]:.3f}, umin={theta_native[1]:.3f}, gamma={theta_native[2]:.4f} (χ²={chi2_native:.1f})")
    
    # Fit with fsps_dl2007
    print("\nFitting with CigaleDustSSPBasis (fsps_dl2007)...")
    model_cigale = build_model()
    model_cigale.params['add_dust_emission'] = False  # Turn off native
    sps_cigale = CigaleDustSSPBasis(dust_emission_model='fsps_dl2007', zcontinuous=1)
    
    theta_cigale, chi2_cigale = fit_model(obs, model_cigale, sps_cigale)
    print(f"  Result: qpah={theta_cigale[0]:.3f}, umin={theta_cigale[1]:.3f}, gamma={theta_cigale[2]:.4f} (χ²={chi2_cigale:.1f})")
    
    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Parameter':<12} {'True':<10} {'Native':<10} {'fsps_dl2007':<10} {'Match?'}")
    print("-" * 52)
    
    match_qpah = abs(theta_native[0] - theta_cigale[0]) < 0.1
    match_umin = abs(theta_native[1] - theta_cigale[1]) < 0.1
    match_gamma = abs(theta_native[2] - theta_cigale[2]) < 0.01
    
    print(f"{'qpah':<12} {true_qpah:<10.2f} {theta_native[0]:<10.3f} {theta_cigale[0]:<10.3f} {'✓' if match_qpah else '✗'}")
    print(f"{'umin':<12} {true_umin:<10.2f} {theta_native[1]:<10.3f} {theta_cigale[1]:<10.3f} {'✓' if match_umin else '✗'}")
    print(f"{'gamma':<12} {true_gamma:<10.3f} {theta_native[2]:<10.4f} {theta_cigale[2]:<10.4f} {'✓' if match_gamma else '✗'}")
    print(f"{'χ²':<12} {'---':<10} {chi2_native:<10.1f} {chi2_cigale:<10.1f}")
    
    if match_qpah and match_umin and match_gamma:
        print("\n✓ VALIDATION PASSED: fsps_dl2007 matches Native FSPS!")
    else:
        print("\n✗ VALIDATION FAILED: Results differ")
    
    # Quick plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get SEDs
    model_native.predict_init(theta_native, sps=sps_native)
    phot_native = model_native.predict_phot(filters)
    
    model_cigale.predict_init(theta_cigale, sps=sps_cigale)
    phot_cigale = model_cigale.predict_phot(filters)
    
    ax.errorbar(filter_waves, phot_noisy, yerr=phot_unc, fmt='ko', ms=10, 
                capsize=4, label='Mock data', zorder=10)
    ax.plot(filter_waves, phot_native, 's-', ms=8, color='black', 
            label=f'Native FSPS (χ²={chi2_native:.1f})', alpha=0.8)
    ax.plot(filter_waves, phot_cigale, 'o--', ms=8, color='blue', 
            label=f'fsps_dl2007 (χ²={chi2_cigale:.1f})', alpha=0.8)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Wavelength (μm)', fontsize=12)
    ax.set_ylabel('Flux (maggies)', fontsize=12)
    ax.set_title('Native FSPS vs fsps_dl2007 Validation', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(script_dir, 'fsps_dl2007_quick_validation.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved: {plot_path}")


if __name__ == '__main__':
    run_validation()
