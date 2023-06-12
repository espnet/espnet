#/bin/bash

# Create symbolic links of tools in prefix/bin to the host-tool variant
set -x

echo "install binutils symlinks ..."
CHOST="${ctng_triplet}"
LDDEPS="as dwp ld ld.bfd ld.gold gprof"
if [[ "$target_platform" == osx-* ]]; then
  LDDEPS=""
fi

for tool in addr2line ar ${LDDEPS} c++filt elfedit nm objcopy objdump ranlib readelf size strings strip; do
  rm -f $PREFIX/bin/$CHOST-$tool || true
  touch $PREFIX/bin/$CHOST-$tool
  # On s390x dwp and gold support seems not to be present
  ln -s $PREFIX/bin/$CHOST-$tool $PREFIX/bin/$tool || true
done

# Gold support is not present on s390x, or on osx
rm -f "$PREFIX/bin/gold" || true
ln -s "$PREFIX/bin/$CHOST-ld.gold" "$PREFIX/bin/gold" || true

