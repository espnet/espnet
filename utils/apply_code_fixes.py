#!/usr/bin/env python3
"""
Script to automatically fix common flake8 and pycodestyle issues in Python files.

This script uses tools like autopep8, black, and isort to automatically
fix common flake8 and pycodestyle issues in Python files. It can also
directly analyze pycodestyle output and apply fixes.
"""

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def find_python_files(
    directory: str, exclude_dirs: Optional[List[str]] = None
) -> List[str]:
    """Find all Python files in the given directory.

    Args:
        directory: The directory to search for Python files.
        exclude_dirs: List of directory names to exclude from the search.

    Returns:
        A list of paths to Python files.
    """
    if exclude_dirs is None:
        exclude_dirs = []

    exclude_set = set(exclude_dirs)
    python_files = []

    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_set]

        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    return python_files


def run_pycodestyle(paths: List[str], exclude: Optional[List[str]] = None) -> str:
    """Run pycodestyle on the specified paths and return the output.

    Args:
        paths: List of file or directory paths to check.
        exclude: List of patterns to exclude.

    Returns:
        The output from pycodestyle.
    """
    exclude_str = ",".join(exclude) if exclude else ""
    cmd = ["pycodestyle", "--show-source", "--show-pep8"]

    if exclude_str:
        cmd.extend(["--exclude", exclude_str])

    cmd.extend(paths)

    try:
        result = subprocess.run(
            cmd,
            check=False,  # Don't raise an exception if pycodestyle finds issues
            capture_output=True,
            text=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running pycodestyle: {e}", file=sys.stderr)
        return e.stdout if e.stdout else ""


def parse_pycodestyle_output(output: str) -> Dict[str, List[Dict[str, str]]]:
    """Parse pycodestyle output and group issues by file.

    Args:
        output: The output from pycodestyle.

    Returns:
        A dictionary mapping file paths to lists of issues.
    """
    issues_by_file = defaultdict(list)

    # Pattern to match pycodestyle output lines
    # Example: file.py:123:4: E101 indentation contains mixed spaces and tabs
    pattern = r"^(.+?):(\d+):(\d+): ([A-Z]\d+) (.+)$"

    current_file = None
    current_line = None
    current_column = None
    current_code = None
    current_message = None
    source_lines = []
    pep8_lines = []

    in_source = False
    in_pep8 = False

    for line in output.split("\n"):
        match = re.match(pattern, line)

        if match:
            # If we were processing an issue, save it before starting a new one
            if current_file and current_line and current_code and current_message:
                issues_by_file[current_file].append(
                    {
                        "line": current_line,
                        "column": current_column,
                        "code": current_code,
                        "message": current_message,
                        "source": "\n".join(source_lines),
                        "pep8": "\n".join(pep8_lines),
                    }
                )
                source_lines = []
                pep8_lines = []

            # Start a new issue
            current_file = match.group(1)
            current_line = match.group(2)
            current_column = match.group(3)
            current_code = match.group(4)
            current_message = match.group(5)
            in_source = False
            in_pep8 = False
        elif line.strip() == "":
            # Empty line toggles between source and PEP8 reference
            if in_source:
                in_source = False
                in_pep8 = True
            elif in_pep8:
                in_pep8 = False
        elif line.startswith("    "):  # Source or PEP8 reference
            if in_source:
                source_lines.append(line)
            elif in_pep8:
                pep8_lines.append(line)
            elif not in_pep8 and not source_lines:
                # First indented line after an issue is source
                in_source = True
                source_lines.append(line)

    # Save the last issue
    if current_file and current_line and current_code and current_message:
        issues_by_file[current_file].append(
            {
                "line": current_line,
                "column": current_column,
                "code": current_code,
                "message": current_message,
                "source": "\n".join(source_lines),
                "pep8": "\n".join(pep8_lines),
            }
        )

    return issues_by_file


def run_command(cmd: List[str], file_path: str) -> bool:
    """Run a shell command and return whether it succeeded.

    Args:
        cmd: The command to run.
        file_path: The file being processed (for error reporting).

    Returns:
        True if the command succeeded, False otherwise.
    """
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        print(f"stdout: {e.stdout}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return False


def fix_from_pycodestyle(
    issues_by_file: Dict[str, List[Dict[str, str]]], verbose: bool = False
) -> Tuple[int, int]:
    """Apply fixes based on pycodestyle output.

    Args:
        issues_by_file: Dictionary mapping file paths to lists of issues.
        verbose: Whether to print verbose output.

    Returns:
        A tuple of (number of files fixed, total number of files with issues).
    """
    fixed_files = set()
    total_files = set(issues_by_file.keys())

    for file_path, issues in issues_by_file.items():
        if verbose:
            print(f"Processing {file_path} with {len(issues)} issues...")

        # Group issues by line number for more efficient processing
        issues_by_line = defaultdict(list)
        for issue in issues:
            issues_by_line[int(issue["line"])].append(issue)

        # Read the file
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)
            continue

        # Track if we made any changes to this file
        file_modified = False

        # Process issues by line
        for line_num, line_issues in sorted(issues_by_line.items()):
            if line_num > len(lines):
                print(
                    f"Warning: Line {line_num} is out of range for {file_path}",
                    file=sys.stderr,
                )
                continue

            line_idx = line_num - 1  # Convert to 0-indexed
            original_line = lines[line_idx]
            modified_line = original_line

            for issue in line_issues:
                code = issue["code"]

                # Apply specific fixes based on error code
                if code == "E201":  # Whitespace after '('
                    modified_line = re.sub(r"\(\s+", "(", modified_line)
                elif code == "E202":  # Whitespace before ')'
                    modified_line = re.sub(r"\s+\)", ")", modified_line)
                elif code == "E203":  # Whitespace before ':'
                    modified_line = re.sub(r"\s+:", ":", modified_line)
                elif code == "E211":  # Whitespace before '('
                    modified_line = re.sub(r"\s+\(", "(", modified_line)
                elif code == "E221":  # Multiple spaces before operator
                    modified_line = re.sub(
                        r"\s{2,}([\+\-\*\/\=])", r" \1", modified_line
                    )
                elif code == "E222":  # Multiple spaces after operator
                    modified_line = re.sub(
                        r"([\+\-\*\/\=])\s{2,}", r"\1 ", modified_line
                    )
                elif code == "E223":  # Tab before operator
                    modified_line = re.sub(r"\t+([\+\-\*\/\=])", r" \1", modified_line)
                elif code == "E224":  # Tab after operator
                    modified_line = re.sub(r"([\+\-\*\/\=])\t+", r"\1 ", modified_line)
                elif code == "E225":  # Missing whitespace around operator
                    for op in [
                        r"\+",
                        "-",
                        r"\*",
                        "/",
                        "%",
                        "<",
                        ">",
                        "&",
                        r"\|",
                        r"\^",
                        "<<",
                        ">>",
                        r"\*\*",
                        "//",
                        "!=",
                        "==",
                        "<=",
                        ">=",
                        r"\+=",
                        "-=",
                        r"\*=",
                        "/=",
                        "%=",
                    ]:
                        modified_line = re.sub(
                            r"([^\s])(" + op + r")([^\s])", r"\1 \2 \3", modified_line
                        )
                elif code == "E231":  # Missing whitespace after ','
                    modified_line = re.sub(r",([^\s])", r", \1", modified_line)
                elif code == "E261":  # At least two spaces before inline comment
                    modified_line = re.sub(r"([^\s])(\s*)#", r"\1  #", modified_line)
                elif code == "E262":  # Inline comment should start with '# '
                    modified_line = re.sub(r"#([^\s])", r"# \1", modified_line)
                elif code == "E265":  # Block comment should start with '# '
                    modified_line = re.sub(r"^(\s*)#([^\s])", r"\1# \2", modified_line)
                elif code == "E271":  # Multiple spaces after keyword
                    modified_line = re.sub(
                        r"\b(if|while|for|return|yield|in|and|or|not|is|elif)\s{2,}",
                        r"\1 ",
                        modified_line,
                    )
                elif code == "E272":  # Multiple spaces before keyword
                    modified_line = re.sub(
                        r"\s{2,}(if|while|for|return|yield|in|and|or|not|is|elif)\b",
                        r" \1",
                        modified_line,
                    )
                elif code == "W291":  # Trailing whitespace
                    modified_line = re.sub(r"\s+$", "", modified_line)
                elif code == "W293":  # Blank line contains whitespace
                    if modified_line.strip() == "":
                        modified_line = "\n"

                if verbose and modified_line != original_line:
                    print(f"  Fixed {code} on line {line_num}")

            # Update the line if it was modified
            if modified_line != original_line:
                lines[line_idx] = modified_line
                file_modified = True

        # Write the modified file back
        if file_modified:
            try:
                with open(file_path, "w") as f:
                    f.writelines(lines)
                fixed_files.add(file_path)
                if verbose:
                    print(f"  Saved changes to {file_path}")
            except Exception as e:
                print(f"Error writing to {file_path}: {e}", file=sys.stderr)

    return len(fixed_files), len(total_files)


def fix_file(
    file_path: str,
    use_black: bool = True,
    use_isort: bool = True,
    use_autopep8: bool = True,
    verbose: bool = False,
) -> bool:
    """Fix flake8 issues in a single file.

    Args:
        file_path: Path to the Python file to fix.
        use_black: Whether to use black for formatting.
        use_isort: Whether to use isort for import sorting.
        use_autopep8: Whether to use autopep8 for PEP 8 compliance.
        verbose: Whether to print verbose output.

    Returns:
        True if all fixes were applied successfully, False otherwise.
    """
    success = True

    if verbose:
        print(f"Processing {file_path}...")

    # Run isort to fix import order issues (H301, H306)
    if use_isort:
        if verbose:
            print(f"  Running isort on {file_path}")
        cmd = ["isort", file_path]
        if not run_command(cmd, file_path):
            success = False

    # Run autopep8 to fix spacing and line issues (E231, etc.)
    if use_autopep8:
        if verbose:
            print(f"  Running autopep8 on {file_path}")
        cmd = ["autopep8", "--in-place", "--aggressive", "--aggressive", file_path]
        if not run_command(cmd, file_path):
            success = False

    # Run black as a final formatter (also handles E203, W503)
    if use_black:
        if verbose:
            print(f"  Running black on {file_path}")
        cmd = ["black", "--line-length", "88", file_path]
        if not run_command(cmd, file_path):
            success = False

    return success


def main():
    """Run the script."""
    parser = argparse.ArgumentParser(
        description="Fix common flake8 and pycodestyle issues in Python files."
    )
    parser.add_argument(
        "paths", nargs="+", help="Paths to files or directories to process."
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=["__pycache__", ".git", "venv", "env", ".venv", ".env"],
        help="Directories to exclude from processing.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "pycodestyle", "tools"],
        default="auto",
        help=(
            "Mode of operation: 'auto' uses both methods, 'pycodestyle' uses "
            "pycodestyle output parsing, 'tools' uses external formatting tools."
        ),
    )
    parser.add_argument(
        "--no-black", action="store_true", help="Don't use black for formatting."
    )
    parser.add_argument(
        "--no-isort", action="store_true", help="Don't use isort for import sorting."
    )
    parser.add_argument(
        "--no-autopep8",
        action="store_true",
        help="Don't use autopep8 for PEP 8 compliance.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print verbose output."
    )

    args = parser.parse_args()

    # Check if pycodestyle is installed
    if args.mode in ["auto", "pycodestyle"]:
        try:
            subprocess.run(
                ["which", "pycodestyle"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError:
            print("Error: pycodestyle is not installed. Please install it using pip:")
            print("  pip install pycodestyle")
            if args.mode == "pycodestyle":
                return 1
            print("Falling back to tools-only mode.")
            args.mode = "tools"

    # For tools mode, check if required tools are installed
    required_tools = []
    if args.mode in ["auto", "tools"]:
        if not args.no_black:
            required_tools.append("black")
        if not args.no_isort:
            required_tools.append("isort")
        if not args.no_autopep8:
            required_tools.append("autopep8")

    missing_tools = []
    for tool in required_tools:
        try:
            subprocess.run(
                ["which", tool],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError:
            missing_tools.append(tool)

    if missing_tools and args.mode in ["auto", "tools"]:
        print(
            "Error: The following required tools are not installed: "
            f"{', '.join(missing_tools)}"
        )
        print("Please install them using pip:")
        print(f"  pip install {' '.join(missing_tools)}")
        if args.mode == "tools":
            return 1
        print("Falling back to pycodestyle-only mode.")
        args.mode = "pycodestyle"

    # Process all files based on the selected mode
    success = True

    # Mode: pycodestyle - parse pycodestyle output and apply fixes
    if args.mode in ["auto", "pycodestyle"]:
        if args.verbose:
            print("Running pycodestyle analysis...")

        output = run_pycodestyle(args.paths, args.exclude)

        if output:
            issues_by_file = parse_pycodestyle_output(output)

            if issues_by_file:
                num_fixed, num_total = fix_from_pycodestyle(
                    issues_by_file, args.verbose
                )

                print(
                    f"Fixed issues in {num_fixed}/{num_total} "
                    "files using pycodestyle analysis."
                )

                if num_fixed < num_total:
                    success = False
        else:
            print("No pycodestyle issues found.")

    # Mode: tools - use external formatting tools
    if args.mode in ["auto", "tools"]:
        # Collect all Python files to process
        all_files = []
        for path in args.paths:
            if os.path.isfile(path) and path.endswith(".py"):
                all_files.append(path)
            elif os.path.isdir(path):
                all_files.extend(find_python_files(path, args.exclude))

        if not all_files:
            print("No Python files found to process.")
            return 0

        if args.verbose:
            print(
                f"Found {len(all_files)} "
                "Python files to process with formatting tools."
            )

        # Process all files
        success_count = 0
        for file_path in all_files:
            if fix_file(
                file_path,
                use_black=not args.no_black,
                use_isort=not args.no_isort,
                use_autopep8=not args.no_autopep8,
                verbose=args.verbose,
            ):
                success_count += 1

        print(
            "Successfully processed "
            f"{success_count}/{len(all_files)} files with formatting tools."
        )

        if success_count < len(all_files):
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    # Execution example: python ./utils/apply_code_fixes.py
    # ./espnet2 --exclude build dist --verbose
    sys.exit(main())
