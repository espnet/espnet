import argparse
import os
import re

platform_file = os.path.join("Lib", "platform.py")
get_version_file = os.path.join("Python", "getversion.c")


def patch_platform(msg):
    with open(platform_file, "r") as fh:
        lines = list(fh)

    lines_it = iter(lines)
    with open(platform_file, "w") as fh:
        for line in lines_it:
            fh.write(line)
            if line.startswith("_sys_version_parser"):
                next_line = next(lines_it)
                fh.write(
                    "    r'([\w.+]+)\s*"
                    + "(?:"
                    + re.escape(" " + msg)
                    + ")?"
                    + "\s*'\n"
                )


def patch_get_version(msg):
    with open(get_version_file, "r") as fh:
        content = list(fh)

    lines = iter(content)
    with open(get_version_file, "w") as fh:
        for line in lines:
            if line.strip().startswith("PyOS_snprintf(version, sizeof(version)"):
                fh.write("    PyOS_snprintf(version, sizeof(version),\n")
                fh.write(
                    '        "%.80s ' + msg.replace('"', '\\"') + ' (%.80s) %.80s",\n'
                )
            else:
                fh.write(line)


msg = os.environ.get("python_branding", "<undefined>")
if msg == "<undefined>":
    msg = "| packaged by conda-forge |"

patch_platform(msg)
patch_get_version(msg)
