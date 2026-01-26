#!/usr/bin/env python
"""
Visualize dust emission SEDs across parameter ranges for all dust models.

This script creates:
1. A comparison plot of DL07, DL2014, and THEMIS at the same parameters
2. Individual parameter variation plots for each dust model:
   - DL07 (CIGALE)
   - DL2014 (Draine 2014)
   - Dale (Dale et al. 2014)
   - THEMIS (Jones et al. 2017)
   - Casey2012 (Casey 2012 modified blackbody)
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

import warnings
warnings.filterwarnings('ignore')

# Import all dust template classes
from prospect.sources.dl2007 import DL2007Templates
from prospect.sources.dl2014 import DL2014Templates
from prospect.sources.themis import ThemisTemplates

# Try to import Dale and Casey2012 if available
try:
    from prospect.sources.dale import DaleTemplates
    HAS_DALE = True
except ImportError:
    HAS_DALE = False
    print("Warning: DaleTemplates not available")

try:
    from prospect.sources.casey2012 import Casey2012Templates
    HAS_CASEY = True
except ImportError:
    HAS_CASEY = False
    print("Warning: Casey2012Templates not available")


def plot_model_comparison():
    """
    Compare DL07, DL2014, and THEMIS at the same parameter values.
    """
    print("=" * 70)
    print("Model Comparison: DL07 vs DL2014 vs THEMIS")
    print("=" * 70)

    # Load templates
    dl07 = DL2007Templates()
    dl14 = DL2014Templates()
    themis = ThemisTemplates()

    # Common parameters
    umin = 1.0
    gamma = 0.1
    qpah = 2.5  # for DL07 and DL2014
    alpha = 2.0  # for DL2014 and THEMIS
    qhac = 0.17  # for THEMIS (mid-range)

    # Get templates
    wave_dl07, spec_dl07, _ = dl07.get_template(qpah=qpah, umin=umin, umax=1e6, gamma=gamma)
    wave_dl14, spec_dl14, _ = dl14.get_template(qpah=qpah, umin=umin, alpha=alpha, gamma=gamma)
    wave_themis, spec_themis, _ = themis.get_template(qhac=qhac, umin=umin, alpha=alpha, gamma=gamma)

    # Convert to microns
    wave_dl07_um = wave_dl07 / 1e4
    wave_dl14_um = wave_dl14 / 1e4
    wave_themis_um = wave_themis / 1e4

    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Dust Emission Model Comparison (Same Parameters)', fontsize=14)

    # Top left: Full SED comparison
    ax = axes[0, 0]
    ax.loglog(wave_dl07_um, spec_dl07 * wave_dl07, 'b-', lw=2, label='DL2007 (CIGALE)', alpha=0.8)
    ax.loglog(wave_dl14_um, spec_dl14 * wave_dl14, 'g-', lw=2, label='DL2014', alpha=0.8)
    ax.loglog(wave_themis_um, spec_themis * wave_themis, 'r-', lw=2, label='THEMIS', alpha=0.8)
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel(r'$\lambda F_\lambda$ (normalized)')
    ax.set_xlim(1, 1000)
    ax.set_title(f'Full SED (umin={umin}, gamma={gamma}, qpah={qpah}, alpha={alpha})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top right: PAH region zoom
    ax = axes[0, 1]
    pah_mask_dl07 = (wave_dl07_um > 2) & (wave_dl07_um < 20)
    pah_mask_dl14 = (wave_dl14_um > 2) & (wave_dl14_um < 20)
    pah_mask_themis = (wave_themis_um > 2) & (wave_themis_um < 20)

    ax.plot(wave_dl07_um[pah_mask_dl07], (spec_dl07 * wave_dl07)[pah_mask_dl07], 'b-', lw=2, label='DL2007', alpha=0.8)
    ax.plot(wave_dl14_um[pah_mask_dl14], (spec_dl14 * wave_dl14)[pah_mask_dl14], 'g-', lw=2, label='DL2014', alpha=0.8)
    ax.plot(wave_themis_um[pah_mask_themis], (spec_themis * wave_themis)[pah_mask_themis], 'r-', lw=2, label='THEMIS', alpha=0.8)
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel(r'$\lambda F_\lambda$ (normalized)')
    ax.set_title('PAH Region (2-20 μm)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom left: FIR peak region
    ax = axes[1, 0]
    fir_mask_dl07 = (wave_dl07_um > 30) & (wave_dl07_um < 500)
    fir_mask_dl14 = (wave_dl14_um > 30) & (wave_dl14_um < 500)
    fir_mask_themis = (wave_themis_um > 30) & (wave_themis_um < 500)

    ax.plot(wave_dl07_um[fir_mask_dl07], (spec_dl07 * wave_dl07)[fir_mask_dl07], 'b-', lw=2, label='DL2007', alpha=0.8)
    ax.plot(wave_dl14_um[fir_mask_dl14], (spec_dl14 * wave_dl14)[fir_mask_dl14], 'g-', lw=2, label='DL2014', alpha=0.8)
    ax.plot(wave_themis_um[fir_mask_themis], (spec_themis * wave_themis)[fir_mask_themis], 'r-', lw=2, label='THEMIS', alpha=0.8)
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel(r'$\lambda F_\lambda$ (normalized)')
    ax.set_title('FIR Peak Region (30-500 μm)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom right: Ratios relative to DL07
    ax = axes[1, 1]
    # Interpolate DL14 and THEMIS onto DL07 wavelength grid
    dl14_interp = np.interp(wave_dl07, wave_dl14, spec_dl14, left=0, right=0)
    themis_interp = np.interp(wave_dl07, wave_themis, spec_themis, left=0, right=0)

    ratio_dl14 = dl14_interp / spec_dl07
    ratio_themis = themis_interp / spec_dl07

    # Mask where DL07 is very small
    valid = spec_dl07 > 1e-20

    ax.semilogx(wave_dl07_um[valid], ratio_dl14[valid], 'g-', lw=1.5, label='DL2014/DL07', alpha=0.8)
    ax.semilogx(wave_dl07_um[valid], ratio_themis[valid], 'r-', lw=1.5, label='THEMIS/DL07', alpha=0.8)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax.fill_between([1, 1000], 0.8, 1.2, alpha=0.1, color='gray')
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('Ratio to DL2007')
    ax.set_title('Template Ratios')
    ax.set_xlim(1, 1000)
    ax.set_ylim(0, 3)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'dust_model_comparison.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")
    plt.close()


def plot_dl07_parameter_variation():
    """Plot DL2007 templates across parameter ranges."""
    print("\n" + "=" * 70)
    print("DL2007 (CIGALE) Parameter Variation")
    print("=" * 70)

    dl07 = DL2007Templates()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DL2007 (CIGALE) Dust Emission Templates', fontsize=14)

    # Top left: Vary qpah
    ax = axes[0, 0]
    qpah_values = [0.47, 1.12, 2.50, 4.58, 7.32]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(qpah_values)))
    for qpah, color in zip(qpah_values, colors):
        wave, spec, _ = dl07.get_template(qpah=qpah, umin=1.0, umax=1e6, gamma=0.1)
        ax.loglog(wave/1e4, spec * wave, '-', color=color, lw=1.5, label=f'qpah={qpah}', alpha=0.8)
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel(r'$\lambda F_\lambda$ (normalized)')
    ax.set_title('Varying qpah (umin=1, gamma=0.1)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top right: Vary umin
    ax = axes[0, 1]
    umin_values = [0.1, 0.5, 1.0, 5.0, 25.0]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(umin_values)))
    for umin, color in zip(umin_values, colors):
        wave, spec, _ = dl07.get_template(qpah=2.5, umin=umin, umax=1e6, gamma=0.1)
        ax.loglog(wave/1e4, spec * wave, '-', color=color, lw=1.5, label=f'umin={umin}', alpha=0.8)
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel(r'$\lambda F_\lambda$ (normalized)')
    ax.set_title('Varying Umin (qpah=2.5, gamma=0.1)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom left: Vary gamma
    ax = axes[1, 0]
    gamma_values = [0.0, 0.01, 0.05, 0.1, 0.5]
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(gamma_values)))
    for gamma, color in zip(gamma_values, colors):
        wave, spec, _ = dl07.get_template(qpah=2.5, umin=1.0, umax=1e6, gamma=gamma)
        ax.loglog(wave/1e4, spec * wave, '-', color=color, lw=1.5, label=f'gamma={gamma}', alpha=0.8)
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel(r'$\lambda F_\lambda$ (normalized)')
    ax.set_title('Varying gamma (qpah=2.5, umin=1)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom right: Parameter grid info
    ax = axes[1, 1]
    ax.axis('off')
    info_text = f"""DL2007 (CIGALE) Template Grid:

qpah values: {dl07.qpah_values}
umin values: {dl07.umin_values[:10]}... (total {len(dl07.umin_values)})
umax values: {dl07.umax_values}

Physical meaning:
- qpah: PAH mass fraction (%)
- Umin: Minimum radiation field intensity
- Umax: Maximum radiation field intensity
- gamma: Fraction of dust in PDR (high-U) regions

Template combination:
spectrum = (1-gamma) * minmin + gamma * minmax
where minmin = delta function at Umin
      minmax = power-law U distribution
"""
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plot_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'dl2007_parameter_variation.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")
    plt.close()


def plot_dl2014_parameter_variation():
    """Plot DL2014 templates across parameter ranges."""
    print("\n" + "=" * 70)
    print("DL2014 (Draine 2014) Parameter Variation")
    print("=" * 70)

    dl14 = DL2014Templates()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DL2014 (Draine 2014) Dust Emission Templates', fontsize=14)

    # Top left: Vary qpah
    ax = axes[0, 0]
    qpah_values = [0.47, 1.12, 2.50, 4.58, 7.32]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(qpah_values)))
    for qpah, color in zip(qpah_values, colors):
        wave, spec, _ = dl14.get_template(qpah=qpah, umin=1.0, alpha=2.0, gamma=0.1)
        ax.loglog(wave/1e4, spec * wave, '-', color=color, lw=1.5, label=f'qpah={qpah}', alpha=0.8)
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel(r'$\lambda F_\lambda$ (normalized)')
    ax.set_title('Varying qpah (umin=1, alpha=2, gamma=0.1)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top right: Vary umin
    ax = axes[0, 1]
    umin_values = [0.1, 0.5, 1.0, 5.0, 25.0]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(umin_values)))
    for umin, color in zip(umin_values, colors):
        wave, spec, _ = dl14.get_template(qpah=2.5, umin=umin, alpha=2.0, gamma=0.1)
        ax.loglog(wave/1e4, spec * wave, '-', color=color, lw=1.5, label=f'umin={umin}', alpha=0.8)
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel(r'$\lambda F_\lambda$ (normalized)')
    ax.set_title('Varying Umin (qpah=2.5, alpha=2, gamma=0.1)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom left: Vary alpha
    ax = axes[1, 0]
    alpha_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(alpha_values)))
    for alpha, color in zip(alpha_values, colors):
        wave, spec, _ = dl14.get_template(qpah=2.5, umin=1.0, alpha=alpha, gamma=0.1)
        ax.loglog(wave/1e4, spec * wave, '-', color=color, lw=1.5, label=f'alpha={alpha}', alpha=0.8)
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel(r'$\lambda F_\lambda$ (normalized)')
    ax.set_title('Varying alpha (qpah=2.5, umin=1, gamma=0.1)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom right: Vary gamma
    ax = axes[1, 1]
    gamma_values = [0.0, 0.01, 0.05, 0.1, 0.5]
    colors = plt.cm.cividis(np.linspace(0.1, 0.9, len(gamma_values)))
    for gamma, color in zip(gamma_values, colors):
        wave, spec, _ = dl14.get_template(qpah=2.5, umin=1.0, alpha=2.0, gamma=gamma)
        ax.loglog(wave/1e4, spec * wave, '-', color=color, lw=1.5, label=f'gamma={gamma}', alpha=0.8)
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel(r'$\lambda F_\lambda$ (normalized)')
    ax.set_title('Varying gamma (qpah=2.5, umin=1, alpha=2)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'dl2014_parameter_variation.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")
    plt.close()


def plot_themis_parameter_variation():
    """Plot THEMIS templates across parameter ranges."""
    print("\n" + "=" * 70)
    print("THEMIS (Jones et al. 2017) Parameter Variation")
    print("=" * 70)

    themis = ThemisTemplates()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('THEMIS Dust Emission Templates', fontsize=14)

    # Top left: Vary qhac
    ax = axes[0, 0]
    qhac_values = themis.qhac_values[::2]  # Every other value
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(qhac_values)))
    for qhac, color in zip(qhac_values, colors):
        wave, spec, _ = themis.get_template(qhac=qhac, umin=1.0, alpha=2.0, gamma=0.1)
        ax.loglog(wave/1e4, spec * wave, '-', color=color, lw=1.5, label=f'qhac={qhac:.2f}', alpha=0.8)
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel(r'$\lambda F_\lambda$ (normalized)')
    ax.set_title('Varying qhac (umin=1, alpha=2, gamma=0.1)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top right: Vary umin
    ax = axes[0, 1]
    umin_values = [0.1, 0.5, 1.0, 5.0, 25.0]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(umin_values)))
    for umin, color in zip(umin_values, colors):
        wave, spec, _ = themis.get_template(qhac=0.17, umin=umin, alpha=2.0, gamma=0.1)
        ax.loglog(wave/1e4, spec * wave, '-', color=color, lw=1.5, label=f'umin={umin}', alpha=0.8)
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel(r'$\lambda F_\lambda$ (normalized)')
    ax.set_title('Varying Umin (qhac=0.17, alpha=2, gamma=0.1)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom left: Vary alpha
    ax = axes[1, 0]
    alpha_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(alpha_values)))
    for alpha, color in zip(alpha_values, colors):
        wave, spec, _ = themis.get_template(qhac=0.17, umin=1.0, alpha=alpha, gamma=0.1)
        ax.loglog(wave/1e4, spec * wave, '-', color=color, lw=1.5, label=f'alpha={alpha}', alpha=0.8)
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel(r'$\lambda F_\lambda$ (normalized)')
    ax.set_title('Varying alpha (qhac=0.17, umin=1, gamma=0.1)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom right: Vary gamma
    ax = axes[1, 1]
    gamma_values = [0.0, 0.01, 0.05, 0.1, 0.5]
    colors = plt.cm.cividis(np.linspace(0.1, 0.9, len(gamma_values)))
    for gamma, color in zip(gamma_values, colors):
        wave, spec, _ = themis.get_template(qhac=0.17, umin=1.0, alpha=2.0, gamma=gamma)
        ax.loglog(wave/1e4, spec * wave, '-', color=color, lw=1.5, label=f'gamma={gamma}', alpha=0.8)
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel(r'$\lambda F_\lambda$ (normalized)')
    ax.set_title('Varying gamma (qhac=0.17, umin=1, alpha=2)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'themis_parameter_variation.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")
    plt.close()


def plot_dale_parameter_variation():
    """Plot Dale templates across parameter ranges."""
    if not HAS_DALE:
        print("\nSkipping Dale - templates not available")
        return

    print("\n" + "=" * 70)
    print("Dale (Dale et al. 2014) Parameter Variation")
    print("=" * 70)

    dale = DaleTemplates()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Dale et al. (2014) Dust Emission Templates', fontsize=14)

    # Left: Vary alpha
    ax = axes[0]
    alpha_values = dale.alpha_values[::2]  # Every other value
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(alpha_values)))
    for alpha, color in zip(alpha_values, colors):
        wave, spec, _ = dale.get_template(alpha=alpha)
        ax.loglog(wave/1e4, spec * wave, '-', color=color, lw=1.5, label=f'alpha={alpha:.2f}', alpha=0.8)
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel(r'$\lambda F_\lambda$ (normalized)')
    ax.set_title('Varying alpha')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: Parameter info
    ax = axes[1]
    ax.axis('off')
    info_text = f"""Dale et al. (2014) Template Grid:

alpha values: {dale.alpha_values}

Physical meaning:
- alpha: Power-law index for dust mass distribution
  dM/dU ~ U^(-alpha)

- Lower alpha: More dust heated to higher U
  (warmer FIR peak, more MIR emission)

- Higher alpha: More dust at lower U
  (cooler FIR peak, less MIR emission)

Note: Dale templates are single-parameter,
simpler than DL07/DL2014/THEMIS but widely used.
"""
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plot_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'dale_parameter_variation.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")
    plt.close()


def plot_casey2012_parameter_variation():
    """Plot Casey2012 templates across parameter ranges."""
    if not HAS_CASEY:
        print("\nSkipping Casey2012 - templates not available")
        return

    print("\n" + "=" * 70)
    print("Casey (2012) Modified Blackbody Parameter Variation")
    print("=" * 70)

    casey = Casey2012Templates()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Casey (2012) Modified Blackbody Templates', fontsize=14)

    # Vary dust temperature
    ax = axes[0, 0]
    T_dust_values = [20, 30, 40, 50, 60]
    colors = plt.cm.hot(np.linspace(0.2, 0.8, len(T_dust_values)))
    for T_dust, color in zip(T_dust_values, colors):
        wave, spec, _ = casey.get_template(T_dust=T_dust, beta=1.8, alpha_MIR=2.0)
        ax.loglog(wave/1e4, spec * wave, '-', color=color, lw=1.5, label=f'T={T_dust}K', alpha=0.8)
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel(r'$\lambda F_\lambda$ (normalized)')
    ax.set_title('Varying T_dust (beta=1.8, alpha_MIR=2)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Vary beta
    ax = axes[0, 1]
    beta_values = [1.0, 1.5, 1.8, 2.0, 2.5]
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(beta_values)))
    for beta, color in zip(beta_values, colors):
        wave, spec, _ = casey.get_template(T_dust=40, beta=beta, alpha_MIR=2.0)
        ax.loglog(wave/1e4, spec * wave, '-', color=color, lw=1.5, label=f'beta={beta}', alpha=0.8)
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel(r'$\lambda F_\lambda$ (normalized)')
    ax.set_title('Varying beta (T=40K, alpha_MIR=2)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Vary alpha_MIR
    ax = axes[1, 0]
    alpha_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(alpha_values)))
    for alpha, color in zip(alpha_values, colors):
        wave, spec, _ = casey.get_template(T_dust=40, beta=1.8, alpha_MIR=alpha)
        ax.loglog(wave/1e4, spec * wave, '-', color=color, lw=1.5, label=f'alpha_MIR={alpha}', alpha=0.8)
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel(r'$\lambda F_\lambda$ (normalized)')
    ax.set_title('Varying alpha_MIR (T=40K, beta=1.8)')
    ax.set_xlim(1, 1000)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Parameter info
    ax = axes[1, 1]
    ax.axis('off')
    info_text = """Casey (2012) Modified Blackbody:

S_nu ~ (1 - exp(-tau)) * B_nu(T_dust) + S_MIR

Parameters:
- T_dust: Dust temperature (K)
- beta: Dust emissivity index (tau ~ nu^beta)
- alpha_MIR: MIR power-law slope

Physical meaning:
- T_dust sets FIR peak position
- beta controls Rayleigh-Jeans slope
- alpha_MIR adds MIR power-law component

Typical values:
- T_dust: 25-50 K for normal galaxies
- beta: 1.5-2.0 (MW-like dust)
- alpha_MIR: 2.0 (typical)
"""
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plot_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'casey2012_parameter_variation.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")
    plt.close()


def main():
    print("=" * 70)
    print("Dust Emission SED Visualization")
    print("=" * 70)

    # 1. Model comparison plot
    plot_model_comparison()

    # 2. Individual model parameter variation plots
    plot_dl07_parameter_variation()
    plot_dl2014_parameter_variation()
    plot_themis_parameter_variation()
    plot_dale_parameter_variation()
    plot_casey2012_parameter_variation()

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print("\nGenerated plots:")
    print("  - dust_model_comparison.png")
    print("  - dl2007_parameter_variation.png")
    print("  - dl2014_parameter_variation.png")
    print("  - themis_parameter_variation.png")
    if HAS_DALE:
        print("  - dale_parameter_variation.png")
    if HAS_CASEY:
        print("  - casey2012_parameter_variation.png")


if __name__ == '__main__':
    main()
