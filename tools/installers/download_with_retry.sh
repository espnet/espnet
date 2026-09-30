#!/usr/bin/env bash
# Sourceable helper. Not executable on its own.
#
#   download_with_retry <url> <output> [extra wget options...]
#
# Downloads a file with backoff and, where there is something cheap to check,
# verifies what came back really is what was asked for. Extracted from
# install_ffmpeg.sh, which had it as a local function, so that ci/install_kaldi.sh
# could stop failing on the same thing:
#
#   Unable to establish SSL connection.
#   ##[error]Process completed with exit code 4.
#
# Extra wget options are forwarded with "$@" rather than through a quoted
# "${var:+...}" scalar, which expands to one empty argument when unset; wget reads
# that as an empty URL, reports "http://: Invalid host name." and exits non-zero
# even when the real download succeeded.
_download_is_intact() {
    case "$1" in
        *.tar | *.tar.* | *.tgz | *.tbz2 | *.txz)
            tar tf "$1" > /dev/null 2>&1
            ;;
        *.zip)
            # unzip is a dependency of the callers that download zips, but do
            # not turn its absence into a download failure.
            if command -v unzip > /dev/null 2>&1; then
                unzip -qt "$1" > /dev/null 2>&1
            else
                [ -s "$1" ]
            fi
            ;;
        *)
            # An installer script or a .deb. There is nothing cheap to verify,
            # but an empty file is still a failure worth retrying.
            [ -s "$1" ]
            ;;
    esac
}

download_with_retry() {
    local url="$1"
    local output="$2"
    shift 2
    local max_attempts=3
    local attempt=1
    local wait=5
    while [ "${attempt}" -le "${max_attempts}" ]; do
        # --tries=1, so every attempt is a fresh connection. wget's own --tries
        # does not recover from a failed TLS handshake, which is how this fails in
        # practice, so --tries=3 alone looks like a retry and is not one.
        if wget "$@" --trust-server-names --tries=1 -O "${output}" "${url}"; then
            # A server can answer 200 with an HTML error page, which would only
            # fail later in tar. Treat that as a failed attempt so a backup URL,
            # where the caller has one, is actually tried. What counts as intact
            # depends on what was asked for - not every download is a tarball.
            if _download_is_intact "${output}"; then
                return 0
            fi
            echo "Downloaded ${output} is not a valid ${output##*.}; the server" \
                 "probably returned an error page"
        fi
        # Leave nothing behind on a failed attempt. wget -O truncates the target
        # before it writes, so a failure otherwise leaves a partial or empty file
        # that a later `[ -e ... ]` guard would happily accept.
        rm -f "${output}"
        echo "Attempt ${attempt}/${max_attempts} failed for ${url}"
        if [ "${attempt}" -lt "${max_attempts}" ]; then
            echo "Waiting ${wait}s before retry..."
            sleep "${wait}"
            wait=$((wait * 2))
        fi
        attempt=$((attempt + 1))
    done
    return 1
}
