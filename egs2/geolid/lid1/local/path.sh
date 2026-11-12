# Check optional recipe dependencies without installing anything when sourced.
for _geolid_module in s3prl transformers lang2vec; do
    if ! python3 -c "import ${_geolid_module}" > /dev/null 2>&1; then
        echo "Error: GeoLID could not import ${_geolid_module}." >&2
        echo "Error: Install it in the active Python environment with:" >&2
        echo "  cd \"${MAIN_ROOT:-../../..}/tools\" && ./installers/install_${_geolid_module}.sh" >&2
        unset _geolid_module
        return 1
    fi
done
unset _geolid_module

# Corpus preparation uses dataset scripts, which datasets 4 removed.
if ! python3 -c '
import sys
import datasets
from packaging.specifiers import SpecifierSet
sys.exit(datasets.__version__ not in SpecifierSet("<4"))
' > /dev/null 2>&1; then
    echo "Error: GeoLID corpus preparation requires an importable datasets<4." >&2
    echo "Error: Install a compatible version in the active Python environment with:" >&2
    echo "  python3 -m pip install 'datasets>=2.16,<4'" >&2
    return 1
fi

if ! python3 -c 'import pycountry' > /dev/null 2>&1; then
    echo "Error: GeoLID corpus preparation could not import pycountry." >&2
    echo "Error: Install it in the active Python environment with:" >&2
    echo "  python3 -m pip install pycountry" >&2
    return 1
fi
