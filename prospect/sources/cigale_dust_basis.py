"""
CigaleDustSSPBasis - Unified SSP basis with configurable dust emission models.

This module provides a single basis class that supports multiple dust emission
models through a parameter-based selection system. This replaces the previous
approach of separate basis classes for each dust model.

Supported dust emission models:
- 'fsps_dl2007': FSPS native D&L 2007 templates (Conroy-modified)
- 'dl2007': CIGALE D&L 2007 templates
- 'dl2014': CIGALE D&L 2014 templates
- 'dale2014': Dale et al. 2014 templates
- 'casey2012': Casey 2012 analytical model
- 'themis': Jones et al. 2017 THEMIS model

The 'fsps_dl2007' option is particularly useful for validation: comparing
Python-based dust emission using FSPS native templates against FSPS's internal
Fortran implementation provides a rigorous test of energy balance correctness.

Example:
    >>> from prospect.sources import CigaleDustSSPBasis
    >>> 
    >>> # Use FSPS native templates for validation against Fortran
    >>> sps = CigaleDustSSPBasis(dust_emission_model='fsps_dl2007')
    >>> 
    >>> # Use CIGALE DL2007 templates
    >>> sps = CigaleDustSSPBasis(dust_emission_model='dl2007')
"""

import numpy as np
from .galaxy_basis import FastStepBasis
from .fake_fsps import add_dust_with_absorption_tracking, add_igm

# Import all template classes
from .dl2007 import DL2007Templates
from .dl2014 import DL2014Templates
from .dale2014 import Dale2014Templates
from .casey2012 import Casey2012Model
from .themis import ThemisTemplates
from .fsps_dl2007 import FSPSDL2007Templates

__all__ = ["CigaleDustSSPBasis"]


# Dictionary mapping model names to (template_class, parameter_names, param_prefix)
DUST_MODELS = {
    'fsps_dl2007': {
        'class': FSPSDL2007Templates,
        'params': ['qpah', 'umin', 'gamma'],
        'prefix': 'duste',  # Use FSPS standard names
    },
    'dl2007': {
        'class': DL2007Templates,
        'params': ['qpah', 'umin', 'umax', 'gamma'],
        'prefix': 'duste',
    },
    'dl2014': {
        'class': DL2014Templates,
        'params': ['qpah', 'umin', 'alpha', 'gamma'],
        'prefix': 'duste',
    },
    'dale2014': {
        'class': Dale2014Templates,
        'params': ['alpha'],
        'prefix': 'duste',
    },
    'casey2012': {
        'class': Casey2012Model,
        'params': ['tdust', 'beta', 'alpha'],
        'prefix': 'duste',
    },
    'themis': {
        'class': ThemisTemplates,
        'params': ['qhac', 'umin', 'alpha', 'gamma'],
        'prefix': 'themis',
    },
}

# Default parameter values for each model
DEFAULT_PARAMS = {
    'fsps_dl2007': {'qpah': 3.5, 'umin': 1.0, 'gamma': 0.01},
    'dl2007': {'qpah': 3.5, 'umin': 1.0, 'umax': 1e6, 'gamma': 0.01},
    'dl2014': {'qpah': 3.5, 'umin': 1.0, 'alpha': 2.0, 'gamma': 0.01},
    'dale2014': {'alpha': 2.0},
    'casey2012': {'tdust': 25.0, 'beta': 1.5, 'alpha': 2.0},
    'themis': {'qhac': 0.17, 'umin': 1.0, 'alpha': 2.0, 'gamma': 0.01},
}


class CigaleDustSSPBasis(FastStepBasis):
    """
    SSP basis with configurable dust emission models.
    
    This class generates galaxy spectra using FSPS for stellar populations
    but replaces FSPS's built-in dust emission with a user-selected model.
    All dust emission is computed via energy balance: the absorbed luminosity
    from dust attenuation is re-emitted using the selected template.
    
    Parameters
    ----------
    dust_emission_model : str, optional
        Which dust model to use. Options:
        - 'fsps_dl2007': FSPS native D&L 2007 templates (default)
        - 'dl2007': CIGALE D&L 2007 templates
        - 'dl2014': CIGALE D&L 2014 templates
        - 'dale2014': Dale et al. 2014 templates
        - 'casey2012': Casey 2012 analytical model
        - 'themis': Jones et al. 2017 THEMIS model
    zcontinuous : int, optional
        The zcontinuous parameter for FSPS (default: 1)
    reserved_params : list, optional
        Additional parameters to reserve from being passed to FSPS
    **kwargs :
        Additional keyword arguments passed to FastStepBasis
    
    Notes
    -----
    Using 'fsps_dl2007' allows direct comparison with FSPS's internal Fortran
    implementation (via FastStepBasis with add_dust_emission=True). This is
    a key validation test: if both approaches give the same result, the
    Python energy balance implementation is verified.
    
    Example
    -------
    >>> from prospect.sources import CigaleDustSSPBasis
    >>> from prospect.models.templates import TemplateLibrary
    >>> 
    >>> # Build model with FSPS DL2007 for validation
    >>> sps = CigaleDustSSPBasis(dust_emission_model='fsps_dl2007')
    >>> 
    >>> # Or use CIGALE DL2014 for different parameterization
    >>> sps = CigaleDustSSPBasis(dust_emission_model='dl2014')
    """
    
    def __init__(self, dust_emission_model='fsps_dl2007', zcontinuous=1,
                 reserved_params=None, **kwargs):
        
        if dust_emission_model not in DUST_MODELS:
            valid = list(DUST_MODELS.keys())
            raise ValueError(f"Unknown dust_emission_model: {dust_emission_model}. "
                           f"Valid options: {valid}")
        
        self.dust_emission_model = dust_emission_model
        model_config = DUST_MODELS[dust_emission_model]
        
        # Build list of reserved parameters (model-specific dust params)
        prefix = model_config['prefix']
        dust_param_names = [f"{prefix}_{p}" for p in model_config['params']]
        
        rp = dust_param_names.copy()
        if reserved_params is not None:
            rp = rp + list(reserved_params)
        
        super().__init__(zcontinuous=zcontinuous, reserved_params=rp, **kwargs)
        
        # Force FSPS to NOT add dust emission - we'll add it ourselves
        self.ssp.params['add_dust_emission'] = False
        
        # Initialize template class (singleton, cached)
        template_class = model_config['class']
        self._templates = template_class()
        self._model_config = model_config
        
        # Store for diagnostics
        self._L_absorbed = 0.0
    
    def _compute_dust_tau(self, wave, dust_type, dust_index, dust2):
        """
        Compute dust optical depth for self-absorption.
        
        This matches the FSPS attenuation curves used for stellar attenuation,
        ensuring self-absorption uses the same dust model.
        
        Parameters
        ----------
        wave : ndarray
            Wavelength in Angstroms
        dust_type : int
            FSPS dust type (0=power-law, 2=Calzetti, 4=Kriek&Conroy, 6=Reddy)
        dust_index : float
            Power-law slope modification
        dust2 : float
            Diffuse dust optical depth at 5500AA
            
        Returns
        -------
        tau : ndarray
            Optical depth at each wavelength
        """
        lamv = 5500.0
        dd63 = 6300.0
        dlam = 350.0
        lamuvb = 2175.0
        
        if dust_type == 0:
            # Power-law attenuation
            tau = dust2 * (wave / lamv) ** dust_index
            
        elif dust_type == 2:
            # Calzetti et al. 2000 attenuation
            cal00 = np.zeros_like(wave)
            gt_dd63 = wave > dd63
            le_dd63 = ~gt_dd63
            if gt_dd63.sum() > 0:
                cal00[gt_dd63] = 1.17 * (-1.857 + 1.04 * (1e4 / wave[gt_dd63])) + 1.78
            if le_dd63.sum() > 0:
                cal00[le_dd63] = 1.17 * (-2.156 + 1.509 * (1e4 / wave[le_dd63]) -
                                         0.198 * (1e4 / wave[le_dd63])**2 +
                                         0.011 * (1e4 / wave[le_dd63])**3) + 1.78
            cal00 = cal00 / 0.44 / 4.05  # R = 4.05
            cal00 = np.clip(cal00, 0.0, np.inf)
            tau = dust2 * cal00
            
        elif dust_type == 4:
            # Kriek & Conroy 2013 attenuation
            cal00 = np.zeros_like(wave)
            gt_dd63 = wave > dd63
            le_dd63 = ~gt_dd63
            if gt_dd63.sum() > 0:
                cal00[gt_dd63] = 1.17 * (-1.857 + 1.04 * (1e4 / wave[gt_dd63])) + 1.78
            if le_dd63.sum() > 0:
                cal00[le_dd63] = 1.17 * (-2.156 + 1.509 * (1e4 / wave[le_dd63]) -
                                         0.198 * (1e4 / wave[le_dd63])**2 +
                                         0.011 * (1e4 / wave[le_dd63])**3) + 1.78
            cal00 = cal00 / 0.44 / 4.05
            cal00 = np.clip(cal00, 0.0, np.inf)
            
            eb = 0.85 - 1.9 * dust_index  # KC13 Eqn 3
            drude = eb * (wave * dlam)**2 / ((wave**2 - lamuvb**2)**2 + (wave * dlam)**2)
            
            tau = dust2 * (cal00 + drude / 4.05) * (wave / lamv) ** dust_index
            
        elif dust_type == 6:
            # Reddy et al. 2015 attenuation
            reddy = np.zeros_like(wave)
            
            w1 = np.abs(wave - 1500).argmin()
            w2 = np.abs(wave - 6000).argmin()
            reddy[w1:w2] = (-5.726 + 4.004 / (wave[w1:w2] / 1e4) - 
                           0.525 / (wave[w1:w2] / 1e4)**2 +
                           0.029 / (wave[w1:w2] / 1e4)**3 + 2.505)
            reddy[:w1] = reddy[w1]  # constant extrapolation blueward
            
            w1 = np.abs(wave - 6000).argmin()
            w2 = np.abs(wave - 28500).argmin()
            reddy[w1:w2] = (-2.672 - 0.010 / (wave[w1:w2] / 1e4) +
                           1.532 / (wave[w1:w2] / 1e4)**2 -
                           0.412 / (wave[w1:w2] / 1e4)**3 + 2.505 - 0.036221981)
            
            reddy = reddy / 2.505
            tau = dust2 * reddy
            
        else:
            # Fallback to power-law with default index for unsupported types
            tau = dust2 * (wave / lamv) ** (-0.7)
            
        return tau
    
    def _get_dust_template(self, wave):
        """
        Get normalized dust emission template for current parameters.
        
        Parameters
        ----------
        wave : ndarray
            Wavelength grid in Angstroms
            
        Returns
        -------
        template : ndarray
            Dust emission template normalized to emit 1 L_sun total
        """
        model = self.dust_emission_model
        prefix = self._model_config['prefix']
        defaults = DEFAULT_PARAMS[model]
        
        match model:
            case 'fsps_dl2007':
                qpah = float(self.params.get(f'{prefix}_qpah', defaults['qpah']))
                umin = float(self.params.get(f'{prefix}_umin', defaults['umin']))
                gamma = float(self.params.get(f'{prefix}_gamma', defaults['gamma']))
                _, template = self._templates.get_template(qpah, umin, gamma, target_wave=wave)
                
            case 'dl2007':
                qpah = float(self.params.get(f'{prefix}_qpah', defaults['qpah']))
                umin = float(self.params.get(f'{prefix}_umin', defaults['umin']))
                umax = float(self.params.get(f'{prefix}_umax', defaults['umax']))
                gamma = float(self.params.get(f'{prefix}_gamma', defaults['gamma']))
                _, template, _ = self._templates.get_template(qpah, umin, umax, gamma, target_wave=wave)
                
            case 'dl2014':
                qpah = float(self.params.get(f'{prefix}_qpah', defaults['qpah']))
                umin = float(self.params.get(f'{prefix}_umin', defaults['umin']))
                alpha = float(self.params.get(f'{prefix}_alpha', defaults['alpha']))
                gamma = float(self.params.get(f'{prefix}_gamma', defaults['gamma']))
                _, template, _ = self._templates.get_template(qpah, umin, alpha, gamma, target_wave=wave)
                
            case 'dale2014':
                alpha = float(self.params.get(f'{prefix}_alpha', defaults['alpha']))
                _, template = self._templates.get_template(alpha, target_wave=wave)
                
            case 'casey2012':
                tdust = float(self.params.get(f'{prefix}_tdust', defaults['tdust']))
                beta = float(self.params.get(f'{prefix}_beta', defaults['beta']))
                alpha = float(self.params.get(f'{prefix}_alpha', defaults['alpha']))
                _, template = self._templates.get_spectrum(tdust, beta, alpha, target_wave=wave)
                
            case 'themis':
                qhac = float(self.params.get(f'{prefix}_qhac', defaults['qhac']))
                umin = float(self.params.get(f'{prefix}_umin', defaults['umin']))
                alpha = float(self.params.get(f'{prefix}_alpha', defaults['alpha']))
                gamma = float(self.params.get(f'{prefix}_gamma', defaults['gamma']))
                _, template, _ = self._templates.get_template(qhac, umin, alpha, gamma, target_wave=wave)
                
            case _:
                raise ValueError(f"Unknown dust emission model: {model}")
        
        return template
    
    def _add_dust_emission_with_self_absorption(self, L_absorbed, wave, dust2=None,
                                                  dust_type=None, dust_index=None):
        """
        Add dust emission with iterative self-absorption (matching FSPS exactly).
        
        The dust emission is attenuated by the same ISM dust that attenuates stars,
        and the absorbed energy is re-emitted. This iterates until convergence
        (< 0.01 L_sun change per iteration), matching FSPS add_dust.f90.
        
        This method can be used for any source of absorbed luminosity (stellar,
        AGN, or combined).
        
        Parameters
        ----------
        L_absorbed : float
            Total absorbed luminosity in L_sun (can be from any source)
        wave : ndarray
            Wavelength grid in Angstroms
        dust2 : float, optional
            Diffuse dust optical depth at 5500A. If None, reads from self.params.
        dust_type : int, optional
            Dust attenuation curve type. If None, reads from self.params/ssp.
        dust_index : int, optional
            Power-law index for dust_type=0. If None, reads from self.params/ssp.
        
        Returns
        -------
        dust_emission : ndarray
            Dust emission spectrum in L_sun/Hz with self-absorption applied
        """
        # Get dust parameters if not provided
        if dust2 is None:
            dust2 = float(self.params.get('dust2', getattr(self.ssp.params, 'dust2', 0.0)))
        if dust_type is None:
            dust_type = int(self.params.get('dust_type', getattr(self.ssp.params, 'dust_type', 0)))
        if dust_index is None:
            dust_index = float(self.params.get('dust_index', getattr(self.ssp.params, 'dust_index', 0.0)))
        
        # Get normalized dust template
        dust_template = self._get_dust_template(wave)
        
        # Scale by absorbed luminosity
        dust_emission = L_absorbed * dust_template
        
        # Apply dust self-absorption (matching FSPS add_dust.f90 exactly)
        if dust2 > 0:
            # Compute diff_dust using same attenuation curve as stellar attenuation
            tau_dust = self._compute_dust_tau(wave, dust_type, dust_index, dust2)
            diff_dust = np.exp(-tau_dust)
            
            # Iterative self-absorption (matching FSPS add_dust.f90 loop exactly)
            # FSPS: DO WHILE (((lboln-lbold).GT.1E-2).OR.iself.EQ.0)
            # Uses ABSOLUTE threshold of 0.01 L_sun
            nu = 2.998e18 / wave
            tduste = np.zeros_like(dust_emission)
            duste = dust_emission.copy()
            iself = 0
            lboln, lbold = 0.0, 0.0
            
            while (lboln - lbold) > 1e-2 or iself == 0:
                oduste = duste.copy()
                duste = duste * diff_dust
                tduste = tduste + duste
                
                lbold = -np.trapezoid(duste, nu)    # after self-absorption
                lboln = -np.trapezoid(oduste, nu)   # before self-absorption
                
                # Re-emit absorbed energy using normalized template shape
                duste = (lboln - lbold) * dust_template
                duste = np.maximum(duste, 1e-30)  # tiny_number
                
                iself = 1
            
            dust_emission = tduste
        
        return dust_emission
    
    def get_galaxy_spectrum(self, **params):
        """
        Generate galaxy spectrum with configurable dust emission.
        
        This method:
        1. Generates a stellar spectrum from FSPS (without dust emission)
        2. Applies dust attenuation, tracking the absorbed luminosity
        3. Adds dust emission from selected model, scaled by absorbed luminosity
        
        Parameters
        ----------
        **params : dict
            Model parameters including stellar population parameters,
            dust attenuation parameters (dust_type, dust2, etc.), and
            dust emission parameters (model-specific)
        
        Returns
        -------
        wave : ndarray
            Wavelength in Angstroms
        spec : ndarray
            Total spectrum (stellar + dust) in L_sun/Hz per Msun formed
        mfrac : float
            Surviving mass fraction
        """
        self.update(**params)
        
        # Determine SFH type based on parameters present
        use_parametric_sfh = 'agebins' not in self.params
        
        if use_parametric_sfh:
            # Parametric SFH (tau model, delayed-tau, etc.)
            # For parametric SFH, sps.get_spectrum() returns spectrum per Msun formed,
            # so mtot=1.0 to avoid double-normalizing
            mtot = 1.0
            tage = float(self.params.get('tage', 1.0))
            
            # Set FSPS params for parametric SFH
            self.ssp.params["sfh"] = int(self.params.get('sfh', 1))  # 1=tau, 4=delayed-tau
            self.ssp.params["tau"] = float(self.params.get('tau', 1.0))
            self.ssp.params["tage"] = tage
            self.ssp.params["add_dust_emission"] = False
            
            # IMPORTANT: Set dust2=0 in FSPS since we handle attenuation externally
            # via add_dust_with_absorption_tracking. Otherwise attenuation is applied twice!
            self.ssp.params["dust2"] = 0.0
            self.ssp.params["dust1"] = 0.0
            
            # Update other FSPS params (but NOT dust2/dust1)
            for key in ['logzsol', 'dust_type', 'dust_index', 'dust1_index', 'fagn', 'agn_tau']:
                if key in self.params:
                    self.ssp.params[key] = float(self.params[key])
            
            wave, spec = self.ssp.get_spectrum(tage=tage, peraa=False)
            
            # For parametric SFH, approximate young/old split
            # (simplified - treat all as "old" for dust purposes)
            young = np.zeros_like(spec)
            old = spec.copy()
            specs = [young, old]
            
        else:
            # Non-parametric SFH (agebins + mass)
            if np.min(np.diff(10**self.params['agebins'])) < 1e6:
                raise ValueError("Agebins must have minimum spacing of 1 Myr")
            
            mtot = self.params['mass'].sum()
            time, sfr, tmax = self.convert_sfh(self.params['agebins'], self.params['mass'])
            
            # Get stellar spectrum WITHOUT dust emission
            self.ssp.params["sfh"] = 3
            self.ssp.params["add_dust_emission"] = False
            self.ssp.set_tabular_sfh(time, sfr)
            
            wave, spec = self.ssp.get_spectrum(tage=tmax, peraa=False)
            
            # Get young/old components
            young, old = self.ssp._csp_young_old
            specs = [young, old]
        
        # Get emission lines
        ewave = self.ssp.emline_wavelengths
        eline_lum = self.ssp.emline_luminosity.copy()
        if eline_lum.ndim > 1:
            eline_lum = eline_lum[0]
        # Assume all lines come from young component
        elines = [eline_lum, np.zeros_like(eline_lum)]
        
        # Apply dust attenuation and track absorbed energy
        dust_type = int(self.params.get('dust_type', self.ssp.params['dust_type']))
        dust_index = float(self.params.get('dust_index', self.ssp.params['dust_index']))
        dust2 = float(self.params.get('dust2', self.ssp.params['dust2']))
        dust1_index = float(self.params.get('dust1_index', self.ssp.params['dust1_index']))
        dust1 = float(self.params.get('dust1', self.ssp.params['dust1']))
        frac_nodust = float(self.params.get('frac_nodust', 0.0))
        frac_obrun = float(self.params.get('frac_obrun', 0.0))
        
        attenuated_spec, attenuated_lines, L_absorbed = add_dust_with_absorption_tracking(
            wave, specs, ewave, elines,
            dust_type=dust_type, dust_index=dust_index, dust2=dust2,
            dust1_index=dust1_index, dust1=dust1,
            frac_nodust=frac_nodust, frac_obrun=frac_obrun
        )
        
        # Apply IGM absorption if requested
        attenuated_spec = add_igm(wave, attenuated_spec, **self.params)
        
        # Add dust emission with self-absorption
        dust_emission = self._add_dust_emission_with_self_absorption(
            L_absorbed, wave, dust2=dust2, dust_type=dust_type, dust_index=dust_index
        )
        
        # Combine stellar + dust emission
        total_spec = attenuated_spec + dust_emission
        
        # Store diagnostics
        self._L_absorbed = L_absorbed / mtot
        self._line_specific_luminosity = attenuated_lines / mtot
        
        # Stellar mass fraction from FSPS
        stellar_mass_frac = self.ssp.stellar_mass / mtot
        
        return wave, total_spec / mtot, stellar_mass_frac
    
    def get_galaxy_elines(self):
        """
        Get attenuated emission line wavelengths and luminosities.
        
        Returns
        -------
        ewave : ndarray
            Emission line wavelengths in Angstroms
        elum : ndarray
            Specific emission line luminosities in L_sun per Msun formed
        """
        ewave = self.ssp.emline_wavelengths
        elum = getattr(self, "_line_specific_luminosity", None)
        
        if elum is None:
            elum = self.ssp.emline_luminosity.copy()
            if elum.ndim > 1:
                elum = elum[0]
            mass = np.sum(self.params.get('mass', 1.0))
            elum /= mass
        
        return ewave, elum
    
    @property
    def L_absorbed(self):
        """Absorbed luminosity per unit mass formed (L_sun/Msun)."""
        return getattr(self, '_L_absorbed', 0.0)
    
    @property
    def L_dust(self):
        """Dust emission luminosity per unit mass formed (L_sun/Msun).
        
        By energy balance, this equals L_absorbed.
        """
        return self.L_absorbed
