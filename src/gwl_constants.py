
import numpy as np

# Intra orbital coulomb interaction
u     =  1.0
# Hund's coupling
j     =  0.0
# Orbital symmetry
isym  =  [[1,2,3,4,5,6],[1,2,3,4,5,6]]
# Smear method
smear =  {'gauss': 0.1}
# Elm & R matrix mix method
mixer =  {'broyden1': [40, 1e-5]}
# Local occupation for double counting
nudc  =  5.0
# Local diagonal basis method
# 0 : original basis
# 1 : automatic diagonal impurity level
# 2 : local diagonal basis transformation given by gwl.ldbs.in
ldbs  =  1
