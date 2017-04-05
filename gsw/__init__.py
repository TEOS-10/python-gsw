# -*- coding: utf-8 -*-

from .gibbs import *
from .utilities import (Bunch, match_args_return, read_data, strip_mask,
                        loadmatbunch)

data = read_data('gsw_data_v3_0.npz')

__version__ = '3.0.6'
__all__ = ['gibbs', 'utilities']
__data_date__ = str(data['version_date'])
__data_version__ = str(data['version_number'])
