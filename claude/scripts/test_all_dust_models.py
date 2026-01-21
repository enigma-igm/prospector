#!/usr/bin/env python
"""
Comprehensive dust model validation test.

Tests ALL dust emission models in CigaleDustSSPBasis:
- Native FSPS (reference)
- fsps_dl2007 (should match native exactly)
- dl2007 (CIGALE Draine & Li 2007)
- dl2014 (CIGALE Draine & Li 2014, with alpha)
- dale2014 (Dale et al. 2014, single alpha parameter)
- casey2012 (Casey 2012 modified blackbody)
- themis (Jones et al. 2017)

Each model has different parameters:
- DL models: qpah, umin, gamma (dl2014 adds alpha)
- dale2014: alpha only
- casey2012: tdust, beta, alpha
- themis: qhac, umin, gamma

Uses prospector's lnprobfn with negative=True for optimization.
Uses IntrinsicSpectrum for full SED visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from scipy.optimize import dual_annealing

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

import warnings
warnings.filterwarnings('ignore')

from sedpy.observate import Filter
from prospect.observation import Photometry, IntrinsicSpectrum
from prospect.models.templates import TemplateLibrary
from prospect.models.sedmodel import SpecModel
from prospect.models.priors import TopHat
from prospect.fitting import lnprobfn
from prospect.sources import SSPBasis
from prospect.sources.cigale_dust_basis import CigaleDustSSPBasis


# =============================================================================
# Model configurations
# =============================================================================
DUST_MODELS = {
    'Native FSPS': {
        'type': 'native',
        'params': ['qpah', 'umin', 'gamma'],
        'bounds': {'qpah': (0.5, 7.0), 'umin': (0.1, 25.0), 'gamma': (0.0, 1.0)},
        'color': 'black',
        'ls': '-',
    },
    'fsps_dl2007': {
        'type': 'cigale',
        'model': 'fsps_dl2007',
        'params': ['qpah', 'umin', 'gamma'],
        'bounds': {'qpah': (0.5, 7.0), 'umin': (0.1, 25.0), 'gamma': (0.0, 1.0)},
        'color': 'blue',
        'ls': '--',
    },
    'dl2007': {
        'type': 'cigale',
        'model': 'dl2007',
        'params': ['qpah', 'umin', 'gamma'],
        'bounds': {'qpah': (0.5, 7.0), 'umin': (0.1, 25.0), 'gamma': (0.0, 1.0)},
        'color': 'green',
        'ls': '-.',
    },
    'dl2014': {
        'type': 'cigale',
        'model': 'dl2014',
        'params': ['qpah', 'umin', 'alpha', 'gamma'],
        'bounds': {'qpah': (0.5, 7.0), 'umin': (0.1, 25.0), 'alpha': (1.0, 3.0), 'gamma': (0.0, 1.0)},
        'color': 'orange',
        'ls': ':',
    },
    'dale2014': {
        'type': 'cigale',
        'model': 'dale2014',
        'params': ['alpha'],
        'bounds': {'alpha': (0.5, 4.0)},
        'color': 'purple',
        'ls': '-',
    },
    'casey2012': {
        'type': 'cigale',
        'model': 'casey2012',
        'params': ['tdust', 'beta', 'alpha'],
        'bounds': {'tdust': (15.0, 60.0), 'beta': (1.0, 2.5), 'alpha': (1.5, 3.0)},
        'color': 'brown',
        'ls': '--',
    },
    'themis': {
        'type': 'cigale',
        'model': 'themis',
        'params': ['qhac', 'umin', 'gamma'],
        'bounds': {'qhac': (0.02, 0.25), 'umin': (0.1, 25.0), 'gamma': (0.0, 1.0)},
        'color': 'red',
        'ls': '-.',
    },
}


def get_filters():
    """Get filter set covering optical to submm."""
    filternames = [
        'sdss_r0', 'twomass_J', 'twomass_Ks',  # Optical/NIR
        'wise_w1', 'wise_w2', 'wise_w3', 'wise_w4',  # WISE
        'spitzer_mips_24', 'spitzer_mips_70',  # Spitzer
        'herschel_pacs_100', 'herschel_pacs_160',  # Herschel PACS
        'herschel_spire_250', 'herschel_spire_350',  # Herschel SPIRE
    ]
    return [Filter(f) for f in filternames]


def build_model(model_name, model_config):
    """Build prospector model for given dust model configuration."""
    model_params = TemplateLibrary["parametric_sfh"]
    
    if model_config['type'] == 'native':
        model_params['add_dust_emission'] = {'N': 1, 'init': True, 'isfree': False}
    else:
        model_params['add_dust_emission'] = {'N': 1, 'init': False, 'isfree': False}
    
    # Add dust emission parameters based on model
    for param in model_config['params']:
        bounds = model_config['bounds'][param]
        init = (bounds[0] + bounds[1]) / 2
        model_params[f'duste_{param}'] = {
            'N': 1, 'init': init, 'isfree': True,
            'prior': TopHat(mini=bounds[0], maxi=bounds[1])
        }
    
    # Fixed SFH parameters
    model_params['tau'] = {'N': 1, 'init': 1.0, 'isfree': False}
    model_params['tage'] = {'N': 1, 'init': 5.0, 'isfree': False}
    model_params['mass'] = {'N': 1, 'init': 1e10, 'isfree': False}
    model_params['logzsol'] = {'N': 1, 'init': 0.0, 'isfree': False}
    model_params['dust2'] = {'N': 1, 'init': 0.3, 'isfree': False}
    model_params['zred'] = {'N': 1, 'init': 0.0, 'isfree': False}
    
    return SpecModel(model_params)


def build_sps(model_config):
    """Build SPS object for given dust model configuration."""
    if model_config['type'] == 'native':
        sps = SSPBasis(zcontinuous=1)
        sps.ssp.params['add_dust_emission'] = True
    else:
        sps = CigaleDustSSPBasis(dust_emission_model=model_config['model'], zcontinuous=1)
    return sps


def optimize_model(observations, model, sps, maxiter=100):
    """Optimize using dual_annealing with prospector's lnprobfn."""
    bounds = []
    for p in model.free_params:
        prior = model.config_dict[p].get('prior')
        if hasattr(prior, 'params') and 'mini' in prior.params:
            bounds.append((prior.params['mini'], prior.params['maxi']))
        else:
            bounds.append((0.01, 10.0))
    
    def objective(theta):
        return lnprobfn(theta, model=model, observations=observations, 
                        sps=sps, negative=True)
    
    result = dual_annealing(objective, bounds, maxiter=maxiter, seed=42)
    return result.x, result.fun


def get_sed_prediction(model, theta, sps, wave_grid_aa):
    """Get full SED using IntrinsicSpectrum."""
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


def run_validation():
    """Run comprehensive dust model validation."""
    print("=" * 70)
    print("COMPREHENSIVE DUST MODEL VALIDATION")
    print("=" * 70)
    
    # Setup
    filters = get_filters()
    filter_waves = np.array([f.wave_effective for f in filters])
    filter_waves_um = filter_waves / 1e4
    
    # Wavelength grid for SED (0.3 - 500 microns)
    wave_grid_aa = np.logspace(np.log10(3000), np.log10(5e6), 500)
    wave_grid_um = wave_grid_aa / 1e4
    
    print(f"\nUsing {len(filters)} filters from {filter_waves_um.min():.1f} to {filter_waves_um.max():.0f} μm")
    
    # =========================================================================
    # Generate mock data with Native FSPS
    # =========================================================================
    print("\n" + "-" * 50)
    print("Generating mock data with Native FSPS...")
    
    true_params = {'qpah': 3.5, 'umin': 1.0, 'gamma': 0.01}
    print(f"True params: {true_params}")
    
    model_mock = build_model('Native FSPS', DUST_MODELS['Native FSPS'])
    sps_mock = build_sps(DUST_MODELS['Native FSPS'])
    
    # Set true parameters
    for p, v in true_params.items():
        model_mock.params[f'duste_{p}'] = v
    
    theta_mock = model_mock.theta.copy()
    model_mock.predict_init(theta_mock, sps=sps_mock)
    phot_true = model_mock.predict_phot(filters)
    sed_true = get_sed_prediction(model_mock, theta_mock, sps_mock, wave_grid_aa)
    
    # Add 5% noise
    np.random.seed(42)
    phot_unc = np.abs(phot_true) * 0.05
    phot_noisy = phot_true + np.random.normal(0, phot_unc)
    
    obs = Photometry(filters=filters, flux=phot_noisy, uncertainty=phot_unc)
    
    # =========================================================================
    # Fit with all models
    # =========================================================================
    results = {}
    
    for model_name, config in DUST_MODELS.items():
        print("\n" + "-" * 50)
        print(f"Fitting with {model_name}...")
        
        model = build_model(model_name, config)
        sps = build_sps(config)
        
        theta_best, neg_lnprob = optimize_model([obs], model, sps, maxiter=50)
        
        # Get predictions
        sed = get_sed_prediction(model, theta_best, sps, wave_grid_aa)
        model.predict_init(theta_best, sps=sps)
        phot = model.predict_phot(filters)
        
        # Store results with parameter values
        param_values = {}
        labels = model.theta_labels()
        for i, p in enumerate(config['params']):
            param_values[p] = theta_best[labels.index(f'duste_{p}')]
        
        results[model_name] = {
            'params': param_values,
            'theta': theta_best,
            'neg_lnprob': neg_lnprob,
            'sed': sed,
            'phot': phot,
            'config': config,
        }
        
        param_str = ', '.join([f"{p}={v:.3f}" for p, v in param_values.items()])
        print(f"   {param_str} (-lnP={neg_lnprob:.1f})")
    
    # =========================================================================
    # Create visualization
    # =========================================================================
    print("\n" + "-" * 50)
    print("Creating visualization...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Panel 1: Full SED comparison (top left, larger)
    ax1 = fig.add_subplot(2, 2, 1)
    
    for model_name, res in results.items():
        config = res['config']
        lw = 2.5 if 'Native' in model_name or 'fsps_dl' in model_name else 1.5
        alpha = 1.0 if 'Native' in model_name or 'fsps_dl' in model_name else 0.7
        ax1.loglog(wave_grid_um, res['sed'] * wave_grid_aa, config['ls'], 
                   color=config['color'], lw=lw, alpha=alpha, label=model_name)
    
    ax1.errorbar(filter_waves_um, phot_noisy * filter_waves, yerr=phot_unc * filter_waves,
                 fmt='ko', ms=6, capsize=2, label='Mock data', zorder=10)
    
    # Auto-scale y-axis
    ymin = np.min(phot_noisy * filter_waves) * 0.3
    ymax = np.max(phot_noisy * filter_waves) * 3
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlim(0.5, 500)
    
    ax1.set_xlabel('Wavelength (μm)', fontsize=11)
    ax1.set_ylabel(r'$\lambda F_\lambda$ (maggies $\times$ Å)', fontsize=11)
    ax1.set_title('Full SED Comparison', fontsize=12)
    ax1.legend(fontsize=8, loc='lower right', ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: IR zoom (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    
    ir_mask = wave_grid_um > 5
    for model_name, res in results.items():
        config = res['config']
        lw = 2.5 if 'Native' in model_name or 'fsps_dl' in model_name else 1.5
        alpha = 1.0 if 'Native' in model_name or 'fsps_dl' in model_name else 0.7
        ax2.loglog(wave_grid_um[ir_mask], res['sed'][ir_mask] * wave_grid_aa[ir_mask], 
                   config['ls'], color=config['color'], lw=lw, alpha=alpha, label=model_name)
    
    ir_filter_mask = filter_waves_um > 5
    ax2.errorbar(filter_waves_um[ir_filter_mask], 
                 phot_noisy[ir_filter_mask] * filter_waves[ir_filter_mask],
                 yerr=phot_unc[ir_filter_mask] * filter_waves[ir_filter_mask],
                 fmt='ko', ms=6, capsize=2, zorder=10)
    
    # Auto-scale
    ir_phot = phot_noisy[ir_filter_mask] * filter_waves[ir_filter_mask]
    ax2.set_ylim(ir_phot.min() * 0.3, ir_phot.max() * 3)
    ax2.set_xlim(5, 500)
    
    ax2.set_xlabel('Wavelength (μm)', fontsize=11)
    ax2.set_ylabel(r'$\lambda F_\lambda$ (maggies $\times$ Å)', fontsize=11)
    ax2.set_title('IR Region (Dust Emission)', fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Residuals (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    
    for model_name, res in results.items():
        config = res['config']
        residual = (res['phot'] - phot_noisy) / phot_unc
        ax3.plot(filter_waves_um, residual, 'o-', color=config['color'], 
                 ms=5, alpha=0.8, label=model_name)
    
    ax3.axhline(0, color='k', ls='--', alpha=0.5)
    ax3.axhline(-2, color='gray', ls=':', alpha=0.5)
    ax3.axhline(2, color='gray', ls=':', alpha=0.5)
    ax3.set_xscale('log')
    
    # Auto-scale residuals
    all_residuals = np.concatenate([((r['phot'] - phot_noisy) / phot_unc) for r in results.values()])
    res_max = min(np.abs(all_residuals).max() * 1.2, 10)
    ax3.set_ylim(-res_max, res_max)
    
    ax3.set_xlabel('Wavelength (μm)', fontsize=11)
    ax3.set_ylabel('Residual (σ)', fontsize=11)
    ax3.set_title('Fit Residuals', fontsize=12)
    ax3.legend(fontsize=7, loc='upper left', ncol=2)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Parameter summary table (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # Create table with model-specific parameters
    col_labels = ['Model', 'Parameters', '-lnP']
    table_data = []
    
    for model_name, res in results.items():
        param_str = ', '.join([f"{p}={v:.2f}" for p, v in res['params'].items()])
        table_data.append([model_name, param_str, f"{res['neg_lnprob']:.1f}"])
    
    # Add true values
    true_str = ', '.join([f"{p}={v:.2f}" for p, v in true_params.items()])
    table_data.append(['TRUE (Native)', true_str, '---'])
    
    table = ax4.table(cellText=table_data, colLabels=col_labels,
                      loc='center', cellLoc='left',
                      colWidths=[0.22, 0.55, 0.13])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    
    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Highlight true values row
    for j in range(len(col_labels)):
        table[(len(table_data), j)].set_facecolor('#E2EFDA')
    
    ax4.set_title('Fitted Parameters', fontsize=12, pad=20)
    
    plt.suptitle('Dust Model Validation: Mock Data from Native FSPS\n'
                 '(fsps_dl2007 should match Native exactly; others will differ)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(script_dir) / 'plots' / 'all_dust_models_validation.png'
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {plot_path}")
    
    # =========================================================================
    # Print summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\nTrue values (Native FSPS): {true_params}")
    print("\n" + "-" * 70)
    
    for model_name, res in results.items():
        param_str = ', '.join([f"{p}={v:.3f}" for p, v in res['params'].items()])
        print(f"{model_name:<15} {param_str:<45} -lnP={res['neg_lnprob']:.1f}")
    
    # Check fsps_dl2007 vs native
    native = results['Native FSPS']
    fsps_dl = results['fsps_dl2007']
    
    print("\n" + "-" * 70)
    if abs(native['neg_lnprob'] - fsps_dl['neg_lnprob']) < 1.0:
        print("✓ VALIDATION PASSED: fsps_dl2007 matches Native FSPS!")
    else:
        print("✗ VALIDATION FAILED: fsps_dl2007 differs from Native FSPS")
    
    return results


if __name__ == '__main__':
    run_validation()
