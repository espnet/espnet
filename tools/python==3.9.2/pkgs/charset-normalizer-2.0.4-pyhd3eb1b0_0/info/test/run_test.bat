



pip check
IF %ERRORLEVEL% NEQ 0 exit /B 1
normalizer --help
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0
