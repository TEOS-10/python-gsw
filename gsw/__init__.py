# -*- coding: utf-8 -*-

from .gibbs import *
from .utilities import (Bunch, match_args_return, read_data, strip_mask,
                        loadmatbunch)

data = read_data("gsw_data_v3_0.npz")

__authors__ = ['Eric Firing', u'Bjørn Ådlandsvik', 'Filipe Fernandes']
__license__ = "MIT"
__version__ = "3.0.3"
__data_version__ = str(data['version_number'])
__data_date__ = str(data['version_date'])
__maintainer__ = "Filipe Fernandes"
__email__ = "ocefpaf@gmail.com"
__status__ = "Production"
__created_ = "14-Jan-2010"
__modified__ = "19-August-2013"
__all__ = ['gibbs', 'utilities']
