
import numpy as np

# Intra orbital coulomb interaction
u     =  1.5
# Hund's coupling
j     =  0.15
# Orbital symmetry
osym  =  [[1,1,1,1]]
# Atomic symmetry
asym  =  [1]
# Smear method
smear =  {'gauss': 0.1}
# Elm & R matrix mix method
mixer =  {'broyden1': [40, 1e-5]}
# Local occupation for double counting
nudc  =  2.0
# Local diagonal basis method
# 0 : original basis
# 1 : automatic diagonal impurity level
# 2 : local diagonal basis transformation given by gwl.ldbs.in
ldbs  =  1
