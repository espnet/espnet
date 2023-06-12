if "%ARCH%"=="32" (
    set OSSL_CONFIGURE=VC-WIN32
) ELSE (
    set OSSL_CONFIGURE=VC-WIN64A
)

REM Configure step
REM
REM Conda currently does not perform prefix replacement on Windows, so
REM OPENSSLDIR cannot (reliably) be used to provide functionality such as a
REM default configuration and standard CA certificates on a per-environment
REM basis.  Given that, we set OPENSSLDIR to a location with extremely limited
REM write permissions to limit the risk of non-privileged users exploiting
REM OpenSSL's engines feature to perform arbitrary code execution attacks
REM against applications that load the OpenSSL DLLs.
REM
set PERL=%BUILD_PREFIX%\Library\bin\perl
%BUILD_PREFIX%\Library\bin\perl configure %OSSL_CONFIGURE% ^
    --prefix=%LIBRARY_PREFIX% ^
    --openssldir="%CommonProgramFiles%\ssl"

if errorlevel 1 exit 1

REM Build step
rem if "%ARCH%"=="64" (
rem     ml64 -c -Foms\uptable.obj ms\uptable.asm
rem     if errorlevel 1 exit 1
rem )

nmake
if errorlevel 1 exit 1

rem nmake -f ms\nt.mak
rem if errorlevel 1 exit 1
rem nmake -f ms\ntdll.mak
rem if errorlevel 1 exit 1

REM Testing step
nmake test
if errorlevel 1 exit 1

REM Install software components only; i.e., skip the HTML docs
nmake install_sw
if errorlevel 1 exit 1

REM Install support files for reference purposes.  (Note that the way we
REM configured OPENSSLDIR above makes these files non-functional.)
nmake install_ssldirs OPENSSLDIR=%LIBRARY_PREFIX%\ssl
if errorlevel 1 exit 1

REM Install step
rem copy out32dll\openssl.exe %PREFIX%\openssl.exe
rem copy out32\ssleay32.lib %LIBRARY_LIB%\ssleay32_static.lib
rem copy out32\libeay32.lib %LIBRARY_LIB%\libeay32_static.lib
rem copy out32dll\ssleay32.lib %LIBRARY_LIB%\ssleay32.lib
rem copy out32dll\libeay32.lib %LIBRARY_LIB%\libeay32.lib
rem copy out32dll\ssleay32.dll %LIBRARY_BIN%\ssleay32.dll
rem copy out32dll\libeay32.dll %LIBRARY_BIN%\libeay32.dll
rem mkdir %LIBRARY_INC%\openssl
rem xcopy /S inc32\openssl\*.* %LIBRARY_INC%\openssl\
