import subprocess

import toml


def uninstall_task_extras(pyproject_path="pyproject.toml"):
    with open(pyproject_path, "r") as f:
        data = toml.load(f)

    optional_deps = data.get("project", {}).get("optional-dependencies", {})
    target_names = {"asr", "tts", "enh", "st", "s2t", "s2st", "spk"}
    task_extras = {
        name: deps for name, deps in optional_deps.items() if name in target_names
    }

    if not task_extras:
        print("No task extras found.")
        return

    packages = set()
    for deps in task_extras.values():
        for dep in deps:
            pkg = dep.split()[0].split("=")[0].split("<")[0].split(">")[0]
            packages.add(pkg)

    print("[*] Uninstalling the following task extras dependencies:")
    for pkg in sorted(packages):
        print(f" - {pkg}")

    if packages:
        subprocess.run(["pip", "uninstall", "-y", *sorted(packages)])
    else:
        print("Nothing to uninstall.")


if __name__ == "__main__":
    uninstall_task_extras()
