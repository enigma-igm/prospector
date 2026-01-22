from .galaxy_basis import *
from .agnssp_basis import *
from .cigale_dust_basis import CigaleDustSSPBasis
from .cigale_duste import NenkovaTorusTemplates, FSPSDL2007Templates

# Old individual basis classes moved to deprecated/
# Use CigaleDustSSPBasis(dust_emission_model='...') instead

__all__ = ["CSPSpecBasis", "SSPBasis",
           "FastStepBasis", "AGNSSPBasis",
           "CigaleDustSSPBasis",
           "NenkovaTorusTemplates", "FSPSDL2007Templates"]
