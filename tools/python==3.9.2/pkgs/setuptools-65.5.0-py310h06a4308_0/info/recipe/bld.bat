set SETUPTOOLS_INSTALL_WINDOWS_SPECIFIC_FILES=0
set SETUPTOOLS_DISABLE_VERSIONED_EASY_INSTALL_SCRIPT=1
set DISTRIBUTE_DISABLE_VERSIONED_EASY_INSTALL_SCRIPT=1

%PYTHON% setup.py install --single-version-externally-managed --record=record.txt
if errorlevel 1 exit 1
