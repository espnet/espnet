import os
import glob
import argparse
import yaml


def get_menubar_recursively(directory):
    menubars = []
    print(f'Scanning directory: {directory}')
    for child in glob.glob(os.path.join(directory, '*')):
        if os.path.isdir(child):
            children = get_menubar_recursively(child)
            if len(children) > 0:
                # menubars[0]['children'].append(children[0]['children'])
                menubars.append({
                    'text': child.lower().split('/')[-1].upper(),
                    'children': get_menubar_recursively(child)
                })
        else:
            if os.path.splitext(child)[1].lower() == '.ipynb':
                menubars.append(f'/{child[len(DOCS):-6]}') # remoce '.ipynb'
            elif os.path.splitext(child)[1].lower() == '.md':
                menubars.append(f'/{child[len(DOCS):-3]}') # remoce '.ipynb'
    return menubars


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True,
                        type=str, help='List of directory')
    args = parser.parse_args()

    navbars = []
    sidebars = []

    DOCS = args.root + '/'

    # 1. Create navBar
    # 1.1. Create guide
    navbars.append({
        'text': "Guide",
        'icon': "book",
        'prefix': f"guide/",
        'children': []
    })

    for doc in glob.glob(f"{DOCS}/guide/**/"):
        doc_name = doc.split("/")[-2]
        navbars[0]['children'].append({
            'text': doc_name,
            'prefix': f"{doc_name}/",
            'children': [f"README.md"] + sorted([
                f"{submodule.split('/')[-2]}/README.md"
                for submodule in glob.glob(f"{doc}/**/")
            ])
        })

    # 1.2. Create Tools
    navbars.append({
        'text': "Tools",
        'icon': "wrench",
        'prefix': f"tools/",
        'children': []
    })

    for doc in glob.glob(f"{DOCS}/tools/**/"):
        doc = doc.split("/")[-2]
        navbars[1]['children'].append({
            'text': doc,
            'prefix': f"{doc}/",
            'children': [f"README.md"]
        })

    # 1.2. Create Notebooks
    navbars.append({
        'text': "Demo",
        'icon': "laptop-code",
        'prefix': f"notebook/",
        'children': []
    })

    for doc in glob.glob(f"{DOCS}/notebook/**/"):
        doc = doc.split("/")[-2]
        navbars[2]['children'].append({
            'text': doc,
            'prefix': f"{doc}/",
            'children': [f"README.md"]
        })

    # 1.2.1. sort
    navbars[1]['children'].sort(key=lambda x: x['text'].lower())
    navbars[2]['children'].sort(key=lambda x: x['text'].lower())

    # 1.3 write navBars.yml
    with open('navbars.yml', 'w', encoding='utf-8') as f:
        yaml.dump(navbars, f, default_flow_style=False)

    # 2. Create sidebars
    # 2.1. Create guide
    sidebars.append({
        "text": "Guide",
        "icon": "book",
        "prefix": "guide/",
        "children": "structure",
    })

    # 2.2. Create Tools
    sidebars.append({
        "text": "Tools",
        "icon": "wrench",
        "prefix": "tools/",
        "children": "structure",
    })

    # 2.3. Create Notebooks
    sidebars.append({
        "text": "Demo",
        "icon": "laptop-code",
        "prefix": "notebook/",
        "children": "structure",
    })

    # 2.3. Write sidebars.yml
    with open('sidebars.yml', 'w', encoding='utf-8') as f:
        yaml.dump(sidebars, f, default_flow_style=False)
