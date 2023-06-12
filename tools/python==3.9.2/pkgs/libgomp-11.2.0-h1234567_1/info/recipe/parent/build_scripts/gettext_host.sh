#!/bin/bash

set -e

. ${RECIPE_DIR}/build_scripts/build_env.sh

case "${HOST}" in
    *linux*)
        exit 0
        ;;
esac

rm -rf "${WDIR}/build/gettext-host"
mkdir "${WDIR}/build/gettext-host"
pushd "${WDIR}/build/gettext-host"

    CFLAGS="-pipe ${HOST_CFLAG}"                      \
    LDFLAGS="${HOST_LDFLAG}"                          \
    bash "${WDIR}/gettext/configure"                  \
        --build=${HOST}                               \
        --host="${HOST}"                              \
        --prefix="${WDIR}/buildtools"                 \
        --enable-static                               \
        --disable-java                                \
        --disable-native-java                         \
        --disable-csharp                              \
        --without-emacs                               \
        --disable-openmp                              \
        --with-included-libxml                        \
        --with-included-gettext                       \
        --with-included-glib                          \
        --with-included-libcroco                      \
        --with-included-libunistring                  \
        --with-libncurses-prefix="${WDIR}/buildtools" \
        --with-libiconv-prefix="${WDIR}/buildtools"   \
        --without-libpth-prefix

    echo "Building gettext ..."
    make

    echo "Installing gettext ..."
    make install

popd

# save space ... clean up
rm -rf "${WDIR}/build/gettext-host"
