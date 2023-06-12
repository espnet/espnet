GFORTRAN=$(${PREFIX}/bin/*-gcc -dumpmachine)-gfortran
FFLAGS="-fopenmp -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -pipe"

cmake \
    -H${SRC_DIR} \
    -Bbuild \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_Fortran_COMPILER=${GFORTRAN} \
    -DCMAKE_Fortran_FLAGS="${FFLAGS}" \
    .
