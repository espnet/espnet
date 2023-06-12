@REM Copyright (C) 2012 Anaconda, Inc
@REM SPDX-License-Identifier: BSD-3-Clause
@REM Test first character and last character of %1 to see if first character is a "
@REM   but the last character isn't.
@REM This was a bug as described in https://github.com/ContinuumIO/menuinst/issues/60
@REM When Anaconda Prompt has the form
@REM   %windir%\system32\cmd.exe "/K" "C:\Users\builder\Miniconda3\Scripts\activate.bat" "C:\Users\builder\Miniconda3"
@REM Rather than the correct
@REM    %windir%\system32\cmd.exe /K ""C:\Users\builder\Miniconda3\Scripts\activate.bat" "C:\Users\builder\Miniconda3""
@REM this solution taken from https://stackoverflow.com/a/31359867
@set "_args1=%1"
@set _args1_first=%_args1:~0,1%
@set _args1_last=%_args1:~-1%
@set _args1_first=%_args1_first:"=+%
@set _args1_last=%_args1_last:"=+%
@set _args1=

@if "%_args1_first%"=="+" if NOT "%_args1_last%"=="+" (
    @CALL "%~dp0..\condabin\conda.bat" activate
    @GOTO :End
)

@REM This may work if there are spaces in anything in %*
@CALL "%~dp0..\condabin\conda.bat" activate %*

:End
@set _args1_first=
@set _args1_last=
