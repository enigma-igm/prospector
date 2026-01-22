#!/usr/bin/env python
"""
Full validation: Fit mock data (from Native FSPS) with all CIGALE dust models.
Visualize full SEDs using IntrinsicSpectrum.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import copy

from sedpy.observate import Filter
from prospect.observation import Photometry, IntrinsicSpectrum
from prospect.models.templates import TemplateLibrary
from prospect.models.sedmodel import SpecModel
from prospect.models.priors import TopHat
from prospect.sources import SSPBasis
from prospect.sources.cigale_duste_basis import CigaleDustSSPBasis


def build_model(dust_model='native'):
    """Build parametric SFH model."""
    model_params = TemplateLibrary["parametric_sfh"]
    model_params['add_dust_emission'] = {'N': 1, 'init': True, 'isfree': False}
    
    # Dust emission params - name depends on model
    if dust_model == 'themis':
        model_params['duste_qhac'] = {'N': 1, 'init': 0.17, 'isfree': True,
                                       'prior': TopHat(mini=0.02, maxi=0.25)}
    else:
        model_params['duste_qpah'] = {'N': 1, 'init': 3.5, 'isfree': True,
                                       'prior': TopHat(mini=0.5, maxi=7.0)}
    model_params['duste_umin'] = {'N': 1, 'init': 1.0, 'isfree': True,
                                   'prior': TopHat(mini=0.1, maxi=25.0)}
    model_params['duste_gamma'] = {'N': 1, 'init': 0.01, 'isfree': True,
                                    'prior': TopHat(mini=0.0, maxi=1.0)}
    
    # Fixed params
    model_params['tau'] = {'N': 1, 'init': 1.0, 'isfree': False}
    model_params['tage'] = {'N': 1, 'init': 5.0, 'isfree': False}
    model_params['mass'] = {'N': 1, 'init': 1e10, 'isfree': False}
    model_params['logzsol'] = {'N': 1, 'init': 0.0, 'isfree': False}
    model_params['dust2'] = {'N': 1, 'init': 0.3, 'isfree': False}
    model_params['zred'] = {'N': 1, 'init': 0.0, 'isfree': False}
    
    return SpecModel(model_params)


def get_filters():
    """Get filter set covering optical to submm."""
    filternames = [
        'sdss_r0', 'sdss_i0', 'sdss_z0',  # Optical
        'twomass_J', 'twomass_H', 'twomass_Ks',  # NIR
        'wise_w1', 'wise_w2', 'wise_w3', 'wise_w4',  # MIR
        'spitzer_mips_24', 'spitzer_mips_70',  # Spitzer
        'herschel_pacs_100', 'herschel_pacs_160',  # Herschel PACS
        'herschel_spire_250', 'herschel_spire_350',  # Herschel SPIRE
    ]
    return [Filter(f) for f in filternames]


def fit_model(obs, model, sps, maxiter=100):
    """Fit model using L-BFGS-B."""
    # Get bounds from priors
    bounds = []
    for p in model.free_params:
        prior = model.config_dict[p].get('prior')
        if hasattr(prior, 'params') and 'mini' in prior.params:
            bounds.append((prior.params['mini'], prior.params['maxi']))
        else:
            bounds.append((0.01, 10.0))
    
    def objective(theta):
        try:
            model.predict_init(theta, sps=sps)
            phot_model = model.predict_phot(obs.filters)
            chi2 = np.sum(((phot_model - obs.flux) / obs.uncertainty)**2)
            return chi2
        except Exception as e:
            return 1e10
    
    # Start from initial values
    x0 = model.theta.copy()
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                     options={'maxiter': maxiter})
    return result.x, result.fun


def get_sed_prediction(model, theta, sps, wave_grid):
    """Get full SED prediction using IntrinsicSpectrum."""
    # Create IntrinsicSpectrum observation
    full_sed_obs = IntrinsicSpectrum(
        wavelength=wave_grid,  # Angstroms
        flux=np.ones_like(wave_grid),
        uncertainty=np.ones_like(wave_grid),
        name='full_sed'
    )
    full_sed_obs.redshift = 0.0
    full_sed_obs.rectify()
    
    # Get prediction
    predictions, _ = model.predict(theta, [full_sed_obs], sps=sps)
    return predictions[0]  # maggies


def run_validation():
    """Run full validation with all dust models."""
    print("=" * 70)
    print("FULL DUST MODEL VALIDATION")
    print("Mock data: Native FSPS | Fit: fsps_dl2007, dl2007, dl2014, themis")
    print("=" * 70)
    
    # Setup
    filters = get_filters()
    filter_waves = np.array([f.wave_effective for f in filters])
    
    # Wavelength grid for SED (0.3 - 500 microns)
    wave_grid = np.logspace(np.log10(3000), np.log10(5e6), 500)  # Angstroms
    
    print(f"\nUsing {len(filters)} filters")
    print(f"SED wavelength grid: {wave_grid.min()/1e4:.2f} - {wave_grid.max()/1e4:.0f} μm")
    
    # True parameters
    true_qpah, true_umin, true_gamma = 3.5, 1.0, 0.01
    print(f"\nTrue params: qpah={true_qpah}, umin={true_umin}, gamma={true_gamma}")
    
    # =========================================================================
    # Generate mock data with Native FSPS
    # =========================================================================
    print("\n" + "-" * 50)
    print("Generating mock data with Native FSPS...")
    
    model_mock = build_model('native')
    sps_mock = SSPBasis(zcontinuous=1)
    sps_mock.ssp.params['add_dust_emission'] = True
    
    theta_true = np.array([true_qpah, true_umin, true_gamma])
    model_mock.predict_init(theta_true, sps=sps_mock)
    phot_true = model_mock.predict_phot(filters)
    
    # Get true SED
    sed_true = get_sed_prediction(model_mock, theta_true, sps_mock, wave_grid)
    
    # Add 5% noise
    np.random.seed(42)
    phot_unc = np.abs(phot_true) * 0.05
    phot_noisy = phot_true + np.random.normal(0, phot_unc)
    
    obs = Photometry(filters=filters, flux=phot_noisy, uncertainty=phot_unc)
    
    # =========================================================================
    # Fit with different dust models
    # =========================================================================
    results = {}
    
    # 1. Native FSPS
    print("\n" + "-" * 50)
    print("1. Fitting with Native FSPS...")
    model_native = build_model('native')
    sps_native = SSPBasis(zcontinuous=1)
    sps_native.ssp.params['add_dust_emission'] = True
    
    theta_native, chi2_native = fit_model(obs, model_native, sps_native)
    sed_native = get_sed_prediction(model_native, theta_native, sps_native, wave_grid)
    model_native.predict_init(theta_native, sps=sps_native)
    phot_native = model_native.predict_phot(filters)
    
    results['Native FSPS'] = {
        'theta': theta_native, 'chi2': chi2_native, 'sed': sed_native,
        'phot': phot_native, 'color': 'black', 'ls': '-', 'labels': model_native.free_params
    }
    print(f"   θ={theta_native}, χ²={chi2_native:.2f}")
    
    # 2. fsps_dl2007 (should match native)
    print("\n" + "-" * 50)
    print("2. Fitting with CigaleDustSSPBasis (fsps_dl2007)...")
    model_fsps = build_model('fsps_dl2007')
    sps_fsps = CigaleDustSSPBasis(dust_emission_model='fsps_dl2007', zcontinuous=1)
    
    theta_fsps, chi2_fsps = fit_model(obs, model_fsps, sps_fsps)
    sed_fsps = get_sed_prediction(model_fsps, theta_fsps, sps_fsps, wave_grid)
    model_fsps.predict_init(theta_fsps, sps=sps_fsps)
    phot_fsps = model_fsps.predict_phot(filters)
    
    results['fsps_dl2007'] = {
        'theta': theta_fsps, 'chi2': chi2_fsps, 'sed': sed_fsps,
        'phot': phot_fsps, 'color': 'blue', 'ls': '--', 'labels': model_fsps.free_params
    }
    print(f"   θ={theta_fsps}, χ²={chi2_fsps:.2f}")
    
    # 3. CIGALE DL2007
    print("\n" + "-" * 50)
    print("3. Fitting with CigaleDustSSPBasis (dl2007)...")
    model_dl2007 = build_model('dl2007')
    sps_dl2007 = CigaleDustSSPBasis(dust_emission_model='dl2007', zcontinuous=1)
    
    theta_dl2007, chi2_dl2007 = fit_model(obs, model_dl2007, sps_dl2007)
    sed_dl2007 = get_sed_prediction(model_dl2007, theta_dl2007, sps_dl2007, wave_grid)
    model_dl2007.predict_init(theta_dl2007, sps=sps_dl2007)
    phot_dl2007 = model_dl2007.predict_phot(filters)
    
    results['CIGALE DL2007'] = {
        'theta': theta_dl2007, 'chi2': chi2_dl2007, 'sed': sed_dl2007,
        'phot': phot_dl2007, 'color': 'green', 'ls': '-.', 'labels': model_dl2007.free_params
    }
    print(f"   θ={theta_dl2007}, χ²={chi2_dl2007:.2f}")
    
    # 4. DL2014
    print("\n" + "-" * 50)
    print("4. Fitting with CigaleDustSSPBasis (dl2014)...")
    model_dl2014 = build_model('dl2014')
    sps_dl2014 = CigaleDustSSPBasis(dust_emission_model='dl2014', zcontinuous=1)
    
    theta_dl2014, chi2_dl2014 = fit_model(obs, model_dl2014, sps_dl2014)
    sed_dl2014 = get_sed_prediction(model_dl2014, theta_dl2014, sps_dl2014, wave_grid)
    model_dl2014.predict_init(theta_dl2014, sps=sps_dl2014)
    phot_dl2014 = model_dl2014.predict_phot(filters)
    
    results['DL2014'] = {
        'theta': theta_dl2014, 'chi2': chi2_dl2014, 'sed': sed_dl2014,
        'phot': phot_dl2014, 'color': 'orange', 'ls': ':', 'labels': model_dl2014.free_params
    }
    print(f"   θ={theta_dl2014}, χ²={chi2_dl2014:.2f}")
    
    # 5. THEMIS
    print("\n" + "-" * 50)
    print("5. Fitting with CigaleDustSSPBasis (themis)...")
    model_themis = build_model('themis')
    sps_themis = CigaleDustSSPBasis(dust_emission_model='themis', zcontinuous=1)
    
    theta_themis, chi2_themis = fit_model(obs, model_themis, sps_themis)
    sed_themis = get_sed_prediction(model_themis, theta_themis, sps_themis, wave_grid)
    model_themis.predict_init(theta_themis, sps=sps_themis)
    phot_themis = model_themis.predict_phot(filters)
    
    results['THEMIS'] = {
        'theta': theta_themis, 'chi2': chi2_themis, 'sed': sed_themis,
        'phot': phot_themis, 'color': 'red', 'ls': '--', 'labels': model_themis.free_params
    }
    print(f"   θ={theta_themis}, χ²={chi2_themis:.2f}")
    
    # =========================================================================
    # Create visualization
    # =========================================================================
    print("\n" + "-" * 50)
    print("Creating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel 1: Full SED comparison
    ax = axes[0, 0]
    wave_um = wave_grid / 1e4
    
    # Plot SEDs (λFλ)
    for name, res in results.items():
        lw = 2.5 if 'Native' in name or 'fsps_dl' in name else 1.5
        alpha = 1.0 if 'Native' in name or 'fsps_dl' in name else 0.8
        ax.loglog(wave_um, res['sed'] * wave_grid, res['ls'], color=res['color'],
                  lw=lw, alpha=alpha, label=f"{name} (χ²={res['chi2']:.1f})")
    
    # Plot data points
    ax.errorbar(filter_waves/1e4, phot_noisy * filter_waves, 
                yerr=phot_unc * filter_waves,
                fmt='ko', ms=8, capsize=3, label='Mock data', zorder=10)
    
    ax.set_xlabel('Wavelength (μm)', fontsize=12)
    ax.set_ylabel(r'$\lambda F_\lambda$ (maggies $\times$ Å)', fontsize=12)
    ax.set_title('Full SED Comparison', fontsize=14)
    ax.set_xlim(0.3, 500)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Panel 2: IR zoom (3-500 μm)
    ax = axes[0, 1]
    ir_mask = wave_um > 3
    
    for name, res in results.items():
        lw = 2.5 if 'Native' in name or 'fsps_dl' in name else 1.5
        alpha = 1.0 if 'Native' in name or 'fsps_dl' in name else 0.8
        ax.loglog(wave_um[ir_mask], res['sed'][ir_mask] * wave_grid[ir_mask], 
                  res['ls'], color=res['color'], lw=lw, alpha=alpha, label=name)
    
    ir_filter_mask = filter_waves/1e4 > 3
    ax.errorbar(filter_waves[ir_filter_mask]/1e4, 
                phot_noisy[ir_filter_mask] * filter_waves[ir_filter_mask],
                yerr=phot_unc[ir_filter_mask] * filter_waves[ir_filter_mask],
                fmt='ko', ms=8, capsize=3, zorder=10)
    
    ax.set_xlabel('Wavelength (μm)', fontsize=12)
    ax.set_ylabel(r'$\lambda F_\lambda$ (maggies $\times$ Å)', fontsize=12)
    ax.set_title('IR Region (Dust Emission)', fontsize=14)
    ax.set_xlim(3, 500)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Photometry residuals
    ax = axes[1, 0]
    
    for name, res in results.items():
        residuals = (phot_noisy - res['phot']) / phot_unc
        ax.plot(filter_waves/1e4, residuals, 'o-', color=res['color'], 
                ms=6, alpha=0.8, label=name)
    
    ax.axhline(0, color='gray', linestyle='--', lw=1)
    ax.axhline(2, color='gray', linestyle=':', lw=0.5, alpha=0.5)
    ax.axhline(-2, color='gray', linestyle=':', lw=0.5, alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Wavelength (μm)', fontsize=12)
    ax.set_ylabel('Residual (σ)', fontsize=12)
    ax.set_title('Fit Residuals', fontsize=14)
    ax.set_ylim(-5, 5)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Results summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = [
        ['Model', 'q param', 'umin', 'gamma', 'χ²', 'χ²/N'],
    ]
    for name, res in results.items():
        q_val = res['theta'][0]
        umin_val = res['theta'][1]
        gamma_val = res['theta'][2]
        chi2_n = res['chi2'] / len(filters)
        table_data.append([name, f'{q_val:.2f}', f'{umin_val:.2f}', 
                          f'{gamma_val:.3f}', f'{res["chi2"]:.1f}', f'{chi2_n:.2f}'])
    
    # Add true values row
    table_data.append(['TRUE', f'{true_qpah:.2f}', f'{true_umin:.2f}', 
                      f'{true_gamma:.3f}', '---', '---'])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.22, 0.13, 0.13, 0.13, 0.13, 0.13])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    # Header styling
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    # True values row styling
    for i in range(len(table_data[0])):
        table[(len(table_data)-1, i)].set_facecolor('#90EE90')
    
    ax.set_title('Results Summary', fontsize=12, pad=20)
    
    plt.suptitle('Dust Model Validation: Mock Data from Native FSPS\n'
                 'Fit with Native FSPS, fsps_dl2007, dl2007, dl2014, themis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/Users/joe/python/prospector/claude/scripts/all_dust_models_sed_validation.png', 
                dpi=150, bbox_inches='tight')
    print("\nSaved: all_dust_models_sed_validation.png")
    
    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\nTrue values: qpah={true_qpah}, umin={true_umin}, gamma={true_gamma}")
    print(f"\n{'Model':<15} {'q param':<10} {'umin':<10} {'gamma':<10} {'χ²':<10}")
    print("-" * 55)
    for name, res in results.items():
        print(f"{name:<15} {res['theta'][0]:<10.3f} {res['theta'][1]:<10.3f} "
              f"{res['theta'][2]:<10.4f} {res['chi2']:<10.2f}")
    
    # Check if fsps_dl2007 matches native
    if abs(results['Native FSPS']['chi2'] - results['fsps_dl2007']['chi2']) < 0.1:
        print("\n✓ fsps_dl2007 matches Native FSPS (as expected)")
    else:
        print("\n✗ WARNING: fsps_dl2007 differs from Native FSPS")


if __name__ == '__main__':
    run_validation()
