

set -ex



test -f ${PREFIX}/lib/libuuid.a
test -f ${PREFIX}/lib/libuuid.so
echo "make linter happy"
exit 0
