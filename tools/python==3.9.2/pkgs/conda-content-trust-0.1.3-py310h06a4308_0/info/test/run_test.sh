

set -ex



pip check
pytest -v tests
conda-content-trust --help
exit 0
