import argparse
from pathlib import Path

import yaml


def _recursive_dirs(p: Path):
    mds = list(p.glob("**/*.md"))
    if all([len(md.relative_to(p).parents) == 1 for md in mds]):
        return [
            {
                "text": p.name,
                "prefix": f"{p.name}/",
                "children": mds,
            }
        ]

    dirs = []
    for c in p.glob("*"):
        dirs += _recursive_dirs(c)
    return dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path, help="List of directory")
    args = parser.parse_args()

    espnet3_dir = args.root / "espnet3"
    espnet3_sections = [
        ("get-started", "Get started"),
        ("config", "Configuration"),
        ("stages", "Stages"),
        ("guides", "Guides"),
        ("core", "Core components"),
        ("contributing", "Contributing"),
    ]
    espnet3_pages = [{"text": "Overview", "link": "index.md"}]
    for dirname, text in espnet3_sections:
        if (espnet3_dir / dirname / "index.md").exists():
            espnet3_pages.append(
                {"text": text, "link": f"{dirname}/index.md"}
            )

    # Create navbar (on the upper side of the page)
    navbars = [
        {
            "text": "ESPnet3",
            "icon": "mdi:graduation-cap",
            "prefix": "espnet3/",
            "children": espnet3_pages,
        },
        {
            "text": "Tutorials",
            "icon": "ic:round-school",
            "prefix": "/",
            "children": [
                {"text": "Full ESPnet installation", "link": "installation.md"},
                {"text": "ESPnet3", "link": "espnet3/get-started/index.md"},
                {"text": "ESPnet2", "link": "espnet2_tutorial.md"},
                {"text": "ESPnet1", "link": "espnet1_tutorial.md"},
                {
                    "text": "Training configurations",
                    "link": "espnet2_training_option.md",
                },
                {"text": "Recipe tips", "link": "tutorial.md"},
                {"text": "Audio formatting", "link": "espnet2_format_wav_scp.md"},
                {"text": "Task class and data input system", "link": "espnet2_task.md"},
                {"text": "Docker", "link": "docker.md"},
                {"text": "Job scheduling system", "link": "parallelization.md"},
                {"text": "Distributed training", "link": "espnet2_distributed.md"},
                {"text": "Document Generation", "link": "document.md"},
            ],
        },
        {
            "text": "Demos",
            "icon": "fa-solid:laptop-code",
            "prefix": "notebook/",
            # Mirrors espnet/notebook's current layout (Demos/, Courses/). The
            # older ESPnet2/, ESPnetEZ/, ESPnet1/ directories no longer exist
            # there, and ESPnet-EZ is being removed from ESPnet altogether.
            "children": [
                {"text": "Roadmap", "link": "README.md"},
                "Demos/",
                "Courses/",
            ],
        },
        {
            "text": "Recipes",
            "icon": "fa-solid:mug-hot",
            "prefix": "recipe/",
            "children": [
                {"text": "What is a recipe template?", "link": "README.md"},
            ]
            + [
                p.name
                for p in sorted(list(args.root.glob("recipe/*")))
                if p.stem != "README" and p.is_file()
            ],
        },
        {
            "text": "Python API",
            "icon": "fa-solid:book",
            "prefix": "guide/",
            "children": [
                {
                    "text": module.name,
                    "prefix": module.name,
                    "children": [
                        {
                            "text": f"{submodule.name}",
                            "link": f"{submodule.name}/",
                        }
                        for submodule in sorted(list(module.glob("*")))
                    ],
                }
                for module in sorted(list(args.root.glob("guide/*")))
            ],
        },
        {
            "text": "Shell API",
            "icon": "fa-solid:wrench",
            "prefix": "tools/",
            "children": [
                {
                    "text": module.name,
                    "link": f"{module.name}/",
                }
                for module in sorted(list(args.root.glob("tools/*")))
            ],
        },
    ]

    with open("navbars.yml", "w", encoding="utf-8") as f:
        yaml.dump(navbars, f, default_flow_style=False)

    # 2. Create sidebar (on the left side of the page)
    sidebars = []
    for nav in navbars:
        item = {
            "text": nav["text"],
            "icon": nav["icon"],
            "prefix": nav["prefix"],
        }
        if nav["text"] in ("Tutorials",):
            item["children"] = nav["children"]
        else:
            item["children"] = "structure"
        sidebars.append(item)

    with open("sidebars.yml", "w", encoding="utf-8") as f:
        yaml.dump(sidebars, f, default_flow_style=False)
