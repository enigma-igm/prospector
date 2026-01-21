"""
Nenkova+2008 AGN torus dust emission templates with energy normalization.

This module provides the NenkovaTorusTemplates class, which wraps the
existing agn_torus() function and normalizes templates to emit 1 L_sun.

This allows for energy-balanced dust emission where:
    torus_emission = L_absorbed_dust4 * normalized_template

Reference:
    Nenkova, M. et al. (2008), ApJ, 685, 147-159
"""

import numpy as np
from .fake_fsps import agn_torus

__all__ = ["NenkovaTorusTemplates"]

# Speed of light in Angstrom/s
C_AA = 2.998e18


class NenkovaTorusTemplates:
    """
    Manager for Nenkova+2008 AGN torus dust emission templates.

    This class wraps the existing agn_torus() function and provides
    normalized templates for energy-balanced dust emission.

    The templates are normalized so that integrating over frequency
    yields 1 L_sun, allowing simple scaling by absorbed luminosity:
        torus_emission = L_absorbed * normalized_template

    Parameters
    ----------
    None

    Example
    -------
    >>> nenkova = NenkovaTorusTemplates()
    >>> wave, spec = nenkova.get_template(agn_tau=50, target_wave=wave_grid)
    >>> torus_emission = L_absorbed_dust4 * spec
    """

    _instance = None  # Singleton instance cache

    def __new__(cls):
        """Use singleton pattern to avoid repeated initialization."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the template manager."""
        if self._initialized:
            return
        self._initialized = True

    def get_template(self, agn_tau, target_wave=None):
        """
        Get Nenkova torus template for given optical depth, normalized to 1 L_sun.

        Parameters
        ----------
        agn_tau : float
            Optical depth of the AGN dust torus [5 - 150].
            Controls the shape of the torus SED.
        target_wave : ndarray, optional
            Target wavelength grid in Angstroms.
            If not provided, uses a default grid spanning 1-1000 microns.

        Returns
        -------
        wave : ndarray
            Wavelength in Angstroms
        spec_normalized : ndarray
            Spectrum normalized to emit 1 L_sun total (in L_sun/Hz units).
            Multiply by L_absorbed to get actual emission.
        """
        # Use default wavelength grid if not provided
        if target_wave is None:
            # Default grid: 1 micron to 1000 microns (1e4 to 1e7 Angstroms)
            target_wave = np.logspace(4, 7, 500)

        # Get raw template shape from existing agn_torus()
        raw_template = agn_torus(target_wave, agn_tau)

        # Integrate over frequency to get total emission
        # Note: raw_template is in arbitrary units (f_nu)
        nu = C_AA / target_wave  # Hz

        # Integrate: L_total = integral of f_nu d_nu
        # Since nu decreases as wave increases, we need negative sign
        L_total = -np.trapz(raw_template, nu)

        # Normalize so integral = 1 L_sun
        if L_total > 0:
            spec_normalized = raw_template / L_total
        else:
            # Template is zero (shouldn't happen for valid agn_tau)
            spec_normalized = np.zeros_like(raw_template)

        return target_wave, spec_normalized

    def get_template_unnormalized(self, agn_tau, target_wave=None):
        """
        Get raw (unnormalized) Nenkova torus template.

        This is a convenience wrapper around agn_torus() for cases
        where the raw template shape is needed.

        Parameters
        ----------
        agn_tau : float
            Optical depth of the AGN dust torus [5 - 150].
        target_wave : ndarray, optional
            Target wavelength grid in Angstroms.

        Returns
        -------
        wave : ndarray
            Wavelength in Angstroms
        spec : ndarray
            Raw spectrum in original units (not normalized)
        """
        if target_wave is None:
            target_wave = np.logspace(4, 7, 500)

        raw_template = agn_torus(target_wave, agn_tau)
        return target_wave, raw_template
