"""
FSPS native Draine & Li 2007 dust emission templates.

This module provides the FSPSDL2007Templates class, which loads and normalizes
the native FSPS DL2007 templates from $SPS_HOME/dust/dustem/.

These templates include Conroy's modifications to the original D&L 2007 models
(e.g., 3.3um PAH reduced by 50%). Use this class when you need consistency
with FSPS's internal dust emission.

For the original CIGALE D&L 2007 templates without modifications, use
DL2007Templates from prospect.sources.dl2007.

Reference:
    Draine, B.T. & Li, A. (2007), ApJ, 657, 810-837
"""

import os
import numpy as np

__all__ = ["FSPSDL2007Templates"]

# Speed of light in Angstrom/s
C_AA = 2.998e18

# FSPS DL2007 qpah values (percentage) corresponding to files 00-60
FSPS_QPAH_VALUES = np.array([0.47, 1.12, 1.77, 2.50, 3.19, 3.90, 4.58])
FSPS_QPAH_FILES = ['00', '10', '20', '30', '40', '50', '60']

# FSPS DL2007 Umin values (22 values, matching sps_vars.f90 uminarr)
# Files have 45 columns: 1 wavelength + 22 pairs of (min, max)
FSPS_UMIN_VALUES = np.array([
    0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1.0, 1.2,
    1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 8.0, 12.0, 15.0, 20.0, 25.0
])


class FSPSDL2007Templates:
    """
    Manager for FSPS native Draine & Li 2007 dust emission templates.
    
    This class loads templates from $SPS_HOME/dust/dustem/DL07_MW3.1_*.dat
    and provides interpolation for arbitrary parameter combinations.
    
    The templates are normalized so that integrating over frequency yields
    1 L_sun, allowing simple scaling by absorbed luminosity:
        dust_emission = L_absorbed * normalized_template
    
    Parameters
    ----------
    None (templates loaded from $SPS_HOME)
    
    Notes
    -----
    The FSPS templates include Conroy's modifications:
    - 3.3um PAH feature reduced by 50%
    - Possible other minor adjustments
    
    For unmodified D&L 2007 templates, use DL2007Templates.
    
    Example
    -------
    >>> fsps_dl = FSPSDL2007Templates()
    >>> wave, spec = fsps_dl.get_template(qpah=3.5, umin=1.0, gamma=0.01)
    >>> dust_emission = L_absorbed * spec
    """
    
    _instance = None  # Singleton instance cache
    
    def __new__(cls):
        """Use singleton pattern to avoid reloading templates."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize and load templates."""
        if self._initialized:
            return
        
        self._load_templates()
        self._initialized = True
    
    def _load_templates(self):
        """Load all FSPS DL2007 template files."""
        sps_home = os.getenv('SPS_HOME')
        if sps_home is None:
            raise EnvironmentError(
                "SPS_HOME environment variable not set. "
                "Cannot locate FSPS dust emission templates."
            )
        
        dust_dir = os.path.join(sps_home, 'dust', 'dustem')
        
        # Load first file to get wavelength grid
        first_file = os.path.join(dust_dir, f'DL07_MW3.1_{FSPS_QPAH_FILES[0]}.dat')
        if not os.path.exists(first_file):
            raise FileNotFoundError(
                f"FSPS DL2007 template not found: {first_file}"
            )
        
        data = np.loadtxt(first_file)
        self.wavelength_um = data[:, 0]  # microns
        self.wavelength = self.wavelength_um * 1e4  # Angstroms
        n_wave = len(self.wavelength)
        n_umin = len(FSPS_UMIN_VALUES)
        n_qpah = len(FSPS_QPAH_VALUES)
        
        # Store qpah and umin values
        self.qpah_values = FSPS_QPAH_VALUES
        self.umin_values = FSPS_UMIN_VALUES
        
        # Templates shape: (n_qpah, n_umin, n_wave) for both min and max
        self._templates_min = np.zeros((n_qpah, n_umin, n_wave))
        self._templates_max = np.zeros((n_qpah, n_umin, n_wave))
        
        # Load all qpah files
        for i_qpah, qpah_file in enumerate(FSPS_QPAH_FILES):
            filepath = os.path.join(dust_dir, f'DL07_MW3.1_{qpah_file}.dat')
            data = np.loadtxt(filepath)
            
            # Columns are: lambda, (Umin_min, Umin_max) pairs for each Umin
            for i_umin in range(n_umin):
                col_min = 1 + 2 * i_umin
                col_max = 2 + 2 * i_umin
                self._templates_min[i_qpah, i_umin, :] = data[:, col_min]
                self._templates_max[i_qpah, i_umin, :] = data[:, col_max]
    
    def get_template(self, qpah, umin, gamma, target_wave=None):
        """
        Get FSPS DL2007 dust emission spectrum for given parameters.
        
        The two-component model is:
            spectrum = (1 - gamma) * model_min + gamma * model_max
        
        where:
            - model_min: dust heated by U = Umin only (delta function)
            - model_max: dust with power-law U distribution from Umin to Umax
        
        Parameters
        ----------
        qpah : float
            PAH mass fraction [0.47 - 4.58]
        umin : float
            Minimum radiation field intensity [0.10 - 25.0]
        gamma : float
            Fraction of dust mass in PDR component [0 - 1]
        target_wave : ndarray, optional
            Target wavelength grid in Angstroms.
            If provided, spectrum is interpolated onto this grid.
        
        Returns
        -------
        wave : ndarray
            Wavelength in Angstroms
        spec_normalized : ndarray
            Spectrum normalized to emit 1 L_sun total (in L_sun/Hz)
        """
        # Find bracketing qpah indices for interpolation
        # FSPS locate() returns i such that arr(i) <= x < arr(i+1)
        # searchsorted with side='right' then -1 gives the same behavior
        qlo = np.searchsorted(self.qpah_values, qpah, side='right') - 1
        qlo = np.clip(qlo, 0, len(self.qpah_values) - 2)
        qhi = qlo + 1
        
        # Compute qpah interpolation weight (dq in FSPS)
        q0 = self.qpah_values[qlo]
        q1 = self.qpah_values[qhi]
        dq = np.clip((qpah - q0) / (q1 - q0), 0.0, 1.0)
        
        # Find bracketing umin indices for interpolation
        ulo = np.searchsorted(self.umin_values, umin, side='right') - 1
        ulo = np.clip(ulo, 0, len(self.umin_values) - 2)
        uhi = ulo + 1
        
        # Compute umin interpolation weight (du in FSPS)
        u0 = self.umin_values[ulo]
        u1 = self.umin_values[uhi]
        du = np.clip((umin - u0) / (u1 - u0), 0.0, 1.0)
        
        # Bi-linear interpolation over qpah and umin (matching FSPS add_dust.f90)
        # dumin = (1-dq)*(1-du)*dustem[:, qlo, ulo_min] + dq*(1-du)*dustem[:, qhi, ulo_min] + ...
        spec_min = ((1-dq) * (1-du) * self._templates_min[qlo, ulo, :] +
                    dq * (1-du) * self._templates_min[qhi, ulo, :] +
                    dq * du * self._templates_min[qhi, uhi, :] +
                    (1-dq) * du * self._templates_min[qlo, uhi, :])
        
        spec_max = ((1-dq) * (1-du) * self._templates_max[qlo, ulo, :] +
                    dq * (1-du) * self._templates_max[qhi, ulo, :] +
                    dq * du * self._templates_max[qhi, uhi, :] +
                    (1-dq) * du * self._templates_max[qlo, uhi, :])
        
        # Two-component model
        spec = (1 - gamma) * spec_min + gamma * spec_max
        
        # Convert from FSPS units (Jy cm2 sr-1 H-1) to L_sun/Hz
        # The FSPS units are per hydrogen atom, need to normalize by integrating
        # For now, we just normalize so integral = 1 L_sun
        nu = C_AA / self.wavelength  # Hz
        integral = -np.trapz(spec, nu)
        
        if integral > 0:
            spec_normalized = spec / integral
        else:
            spec_normalized = spec
        
        # Interpolate to target wavelength if provided
        wave_out = self.wavelength
        if target_wave is not None:
            spec_normalized = np.interp(target_wave, self.wavelength, spec_normalized,
                                        left=0.0, right=0.0)
            wave_out = target_wave
            
            # Re-normalize after interpolation (interpolation doesn't preserve integrals)
            nu_target = C_AA / target_wave
            integral_target = -np.trapz(spec_normalized, nu_target)
            if integral_target > 0:
                spec_normalized = spec_normalized / integral_target
        
        return wave_out, spec_normalized
    
    def get_nearest_indices(self, qpah, umin):
        """
        Get nearest available qpah and umin indices.
        
        Parameters
        ----------
        qpah : float
            Requested qpah value
        umin : float
            Requested umin value
        
        Returns
        -------
        qpah_idx : int
            Index of nearest available qpah
        umin_idx : int
            Index of nearest available umin
        qpah_actual : float
            Actual qpah value
        umin_actual : float
            Actual umin value
        """
        qpah_idx = np.argmin(np.abs(self.qpah_values - qpah))
        umin_idx = np.argmin(np.abs(self.umin_values - umin))
        return (qpah_idx, umin_idx, 
                self.qpah_values[qpah_idx], self.umin_values[umin_idx])
