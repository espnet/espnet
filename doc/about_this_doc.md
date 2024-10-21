
# Developer's Guide to the ESPnet Homepage

This document outlines the process of automatically generating the ESPnet homepage. It provides step-by-step instructions for building the homepage and details the underlying operations during the generation process.

## Building the Homepage

1. **Clone the ESPnet Repository**: Begin by cloning the ESPnet repository from GitHub.

2. **Generate the `activate_python.sh` Script**:
   - You can generate this file by running either `setup_miniforge.sh` or `setup_venv.sh`.
   - Alternatively, create your own virtual environment and manually write the command to activate it in `activate_python.sh`.

3. **Run `activate_python.sh`**: Execute this script to activate the Python environment.

4. **Install the Dependencies**:
   Install the necessary dependencies using the following commands:
   ```
   espnet[all]
   espnet[doc]
   k2
   chainer
   ```

5. **Build the Homepage**:
   Run the following script to generate the homepage:
   ```
   ./ci/doc.sh
   ```

## Key Points

- The homepage is built using VuePress, a static site generator that converts Markdown files into a website.
- The primary function of `ci/doc.sh` is to generate Markdown files for all documentation.

## Step-by-Step Guide to `ci/doc.sh`

1. **`build_and_convert` Function**:
   This function generates documentation for shell scripts by invoking `./doc/usage2rst.sh` on all scripts in the specified directory (`$1`). The `usage2rst.sh` script executes each script with the `--help` option and saves the output as an RST file in the `$2/<shell_name>.rst` directory.

2. **Temporary Files Directory**:
   All temporary files, including RST files, are stored in the `_gen` directory.

3. **`./doc/argparse2rst.py` Script**:
   This script generates documentation for Python tools located in `espnet/bin`, `espnet2/bin`, and `utils/`. These scripts are executable from the command line, so their documentation is separated from the package information.

4. **`./doc/notebook2rst.sh` Script**:
   This script generates the demo section by pulling the notebook repository and converting Jupyter Notebook (`.ipynb`) files into Markdown.

5. **`./doc/members2rst.py` Script**:
   This script generates RST files for all docstrings. It separates out any docstrings for classes or functions that are not class members and excludes private functions (those starting with `_`). The generated RST files are saved in `./_gen/guide`.

6. **Sphinx Build Process**:
   After copying all necessary files to the `_gen` directory, run `sphinx-build` within `_gen`. Running Sphinx directly in the `doc` directory could cause issues, including potential document corruption. Some files, particularly those ending with `_train` (e.g., `espnet2/bin/asr_train.py`), are excluded from the documentation to avoid errors.

7. **VuePress Directory Setup**:
   Copy the Markdown files from the `doc` directory, along with files generated in steps 4 and 6, into the `vuepress/src` directory. This is where VuePress recognizes the pages for the site.

8. **Language Support Adjustment**:
   VuePress doesn’t support some of the programming languages used in code blocks. To address this, we include a command to replace unsupported language codes with equivalent ones.

9. **Generate Navigation Files**:
   Create the `navbar.yml` and `sidebar.yml` files to define the menus displayed at the top and side of the webpage. For more details, refer to the VuePress-Hope documentation on [navbar configuration](https://theme-hope.vuejs.press/config/theme/layout.html#navbar-config) and [sidebar configuration](https://theme-hope.vuejs.press/config/theme/layout.html#sidebar-config).

10. **Finalize the Build**:
    Install Node.js and the necessary dependencies, then build the homepage. To preview the page, comment out the `docs:build` line and uncomment the `docs:dev` line in the script.
