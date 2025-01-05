"""Setup cython code."""

from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    """
        Setup cython code for the monotonic alignment extension.

    This module configures the setup process for the monotonic alignment
    extension, utilizing Cython to compile the core functionality. The
    `build_ext` class is overridden to customize the build process and
    ensure compatibility with NumPy.

    Attributes:
        exts (list): A list of Extension objects to be built.

    Args:
        name (str): The name of the package.
        ext_modules (list): A list of Cython extension modules to be compiled.
        cmdclass (dict): A mapping of command names to command classes.

    Returns:
        None

    Raises:
        None

    Examples:
        To install the package, run:
            python setup.py build_ext --inplace

    Note:
        This setup script requires Cython and setuptools to be installed
        in your Python environment.

    Todo:
        Add additional extensions as needed in the future.
    """

    def finalize_options(self):
        """
                Finalize the options for the build_ext command.

        This method is overridden to ensure that NumPy does not mistakenly believe it
        is still in its setup process. It also appends the NumPy include directory to
        the list of include directories for the build process.

        Attributes:
            None

        Args:
            None

        Returns:
            None

        Raises:
            None

        Examples:
            To use this class, you would typically call it through the setup function
            as shown in the script.

        Note:
            This method modifies the built-in __NUMPY_SETUP__ variable to False, which
            is a specific workaround for certain NumPy installation scenarios.
        """
        _build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


exts = [
    Extension(
        name="core",
        sources=["core.pyx"],
    )
]
setup(
    name="monotonic_align",
    ext_modules=cythonize(exts, language_level=3),
    cmdclass={"build_ext": build_ext},
)
