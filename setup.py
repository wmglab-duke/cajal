from setuptools import setup, Extension

from Cython.Build import cythonize
import numpy as np

ext_modules = cythonize(
    [
        Extension(
            name="cajal.opt.pareto.nds",
            sources=["./cajal/opt/pareto/nds.pyx"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            name="cajal.opt.pareto.ops",
            sources=["./cajal/opt/pareto/ops.pyx"],
            include_dirs=[np.get_include()],
        ),
    ]
)
setup(ext_modules=ext_modules)
