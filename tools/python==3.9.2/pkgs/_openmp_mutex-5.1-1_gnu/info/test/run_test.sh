

set -ex



test -f ${PREFIX}/lib/libgomp.so.1
test `readlink ${PREFIX}/lib/libgomp.so.1` == "libgomp.so.1.0.0"
exit 0
