REM edited from the one from Anaconda

REM Build step
jom -f makefile.msc lib bzip2
if errorlevel 1 exit 1

REM Install step
copy libbz2.lib %LIBRARY_LIB%\libbz2_static.lib || exit 1
REM Some packages expect 'bzip2.lib', so make another copy
copy libbz2.lib %LIBRARY_LIB%\bzip2_static.lib || exit 1
copy bzlib.h %LIBRARY_INC% || exit 1

cl /O2 /Ibzlib.h /Ibzlib_private.h /D_USRDLL /D_WINDLL blocksort.c bzlib.c compress.c crctable.c decompress.c huffman.c randtable.c /LD /Felibbz2.dll /link /DEF:libbz2.def
if errorlevel 1 exit 1

copy libbz2.lib %LIBRARY_LIB% || exit 1
REM Some packages expect 'bzip2.lib', so make copies
copy libbz2.lib %LIBRARY_LIB%\bzip2.lib || exit 1
copy libbz2.dll %LIBRARY_BIN% || exit 1
copy libbz2.dll %LIBRARY_BIN%\bzip2.dll || exit 1
