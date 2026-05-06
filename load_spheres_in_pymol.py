from pymol.cgo import *
from pymol import cmd

obj = [
    COLOR, 1.0000, 0.0000, 0.0000,
    SPHERE, 0.0000, 0.0000, 0.0000, 1.0000,
    COLOR, 0.0000, 1.0000, 0.0000,
    SPHERE, 3.0000, 0.0000, 0.0000, 0.5000,
    COLOR, 0.0000, 0.0000, 1.0000,
    SPHERE, 0.0000, 3.0000, 0.0000, 1.5000,
    COLOR, 1.0000, 1.0000, 0.0000,
    SPHERE, 0.0000, 0.0000, 3.0000, 0.8000,
]

cmd.load_cgo(obj, 'my_spheres')
