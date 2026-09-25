"""Keep this directory's imports from leaking into the rest of the suite.

Two tests here put the recipe directory on ``sys.path`` so that
``src.inference`` and ``src.preprocessor`` resolve the way a real run resolves
them, which is how the recipe's own configs name them. ``src`` is a name other
recipes use for their own code, and ``test/espnet3/publication`` builds packed
bundles that import ``src.*`` of their own; a module object left behind in
``sys.modules`` is what they would get instead. Dropping the entries when this
package's tests are done keeps the leak inside this directory.
"""

import sys

import ami_sot_paths
import pytest


@pytest.fixture(autouse=True)
def _drop_recipe_src_imports():
    """Remove this recipe's ``src`` package and path entry after each test.

    Per test rather than per directory: this directory has no ``__init__.py``,
    so pytest does not treat it as a package and a package-scoped finalizer is
    not guaranteed to run before the next directory's tests.
    """
    yield
    for name in [n for n in sys.modules if n == "src" or n.startswith("src.")]:
        del sys.modules[name]
    recipe = str(ami_sot_paths.RECIPE)
    while recipe in sys.path:
        sys.path.remove(recipe)
