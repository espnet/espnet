"""Generate Code Documentation with LLM.

This script integrates with flake8 to identify classes and
functions missing docstrings, uses a configurable Large Language
Model (LLM) to generate the documentation, and then automatically
inserts the new docstring into the source file.

Dry run:

```bash
# Gemini
flake8 --select D100,D101,D102,D103 . | python doc_generator.py --api-key \
    "YOUR_GEMINI_API_KEY"

# Ollama
flake8 --select D100,D101,D102,D103 . | python doc_generator.py --llm-type ollama \
    --llm-model llama3

# OpenAI V1 compatible
flake8 --select D100,D101,D102,D103 . | python doc_generator.py \
    --llm-type openai-v1 \
    --llm-ip "http://192.168.1.100:8000" \
    --llm-model "gpt-4" \
    --api-key "YOUR_API_KEY"
```

Apply Changes:

```bash
# Gemini
flake8 --select D100,D101,D102,D103 . | python doc_generator.py --api-key \
    "YOUR_GEMINI_API_KEY" --apply-changes

# Ollama
flake8 --select D100,D101,D102,D103 . | python doc_generator.py \
    --llm-type ollama --llm-model llama3 --apply-changes

# OpenAI V1 compatible
flake8 --select D100,D101,D102,D103 . | python doc_generator.py \
    --llm-type openai-v1 \
    --llm-ip "http://192.168.1.100:8000" \
    --llm-model "gpt-4" \
    --api-key "YOUR_API_KEY" \
    --apply-changes
```

"""

import argparse
import ast
import json
import logging
import os
import re
import sys

import requests

# --- Configuration ---
# This is now a fallback. Use the --api-key argument for better practice.
DEFAULT_API_KEY = os.environ.get("MY_LLM_API_KEY", "YOUR_API_KEY_HERE")

# --- Color Definitions for Logging ---


class Colors:
    """ANSI color codes"""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"


def get_code_snippet(full_source, node):
    """Extracts the source code for a given AST node."""
    return ast.get_source_segment(full_source, node)


def _generate_docstring_gemini(code_snippet, element_type, api_key):
    """Handles API call to Google's Gemini."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    )
    system_prompt = (
        "You are an expert Python programmer. Your task is to write a concise, "
        "professional, and accurate docstring for a given piece of Python code. "
        "The docstring should follow the Google Python Style Guide for docstrings. "
        "Only return the complete docstring content itself, without any "
        "surrounding text, markdown, or the original code."
    )
    user_prompt = (
        "Generate a docstring for the following Python "
        f"{element_type.lower()}:\n\n```python\n{code_snippet}\n```"
    )

    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()

    candidate = result.get("candidates", [{}])[0]
    content = candidate.get("content", {}).get("parts", [{}])[0]
    return content.get("text", "Error: Could not extract text from LLM response.")


def _generate_docstring_ollama(code_snippet, element_type, model, ip):
    """Handles API call to an Ollama server."""
    url = f"{ip}/api/chat"
    system_prompt = (
        "You are a helpful expert Python programmer. Your task is to write a "
        "high-quality docstring for a given Python code snippet. Follow the "
        "Google Python Style Guide. Return only the raw docstring content, "
        "without any introduction, conclusion, or markdown code fences."
    )
    user_prompt = (
        "Generate a docstring for the following Python "
        f"{element_type.lower()}:\n\n```python\n{code_snippet}\n```"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "temperature": 0.7,
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()
    return result.get("message", {}).get(
        "content", "Error: Could not extract 'response' from Ollama result."
    )


def _generate_docstring_openai_v1(code_snippet, element_type, model, ip, api_key):
    """Handles API call to an OpenAI v1 compatible server."""
    url = f"{ip}/v1/chat/completions"
    system_prompt = (
        "You are an expert Python programmer. Your task is to write a concise, "
        "professional, and accurate docstring for a given piece of Python code. "
        "The docstring should follow the Google Python Style Guide for docstrings. "
        "Only return the complete docstring content itself, without any surrounding "
        "text, markdown, or the original code."
    )
    user_prompt = (
        "Generate a docstring for the following Python "
        f"{element_type.lower()}:\n\n```python\n{code_snippet}\n```"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()
    return (
        result.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "Error: Could not parse OpenAI response.")
    )


def generate_docstring_with_llm(code_snippet, element_type, args):
    """Main dispatcher for generating docstrings using the selected LLM provider."""

    # Dry run if no API key is provided for services that need one.
    if args.llm_type in ["gemini", "openai-v1"] and (
        not args.api_key or args.api_key == "YOUR_API_KEY_HERE"
    ):
        logging.warning("---")
        logging.warning(
            f"{Colors.YELLOW}SKIPPING API CALL: API Key not configured for llm-type "
            f"'{args.llm_type}'.{Colors.ENDC}"
        )
        logging.warning(
            "Would have generated a docstring for "
            f"this {element_type.lower()}:"
        )
        logging.warning(code_snippet)
        logging.warning("---")
        return f"/* LLM Generation Skipped for {element_type} */"

    try:
        logging.info(
            f"{Colors.GREEN}-> Sending {element_type} to LLM ({args.llm_type}) "
            f"for documentation...{Colors.ENDC}"
        )
        generated_text = ""
        if args.llm_type == "gemini":
            generated_text = _generate_docstring_gemini(
                code_snippet, element_type, args.api_key
            )
        elif args.llm_type == "ollama":
            if not args.llm_ip or not args.llm_model:
                raise ValueError(
                    "For --llm-type 'ollama', you must provide --llm-ip and "
                    "--llm-model."
                )
            generated_text = _generate_docstring_ollama(
                code_snippet, element_type, args.llm_model, args.llm_ip
            )
        elif args.llm_type == "openai-v1":
            if not args.llm_ip or not args.llm_model:
                raise ValueError(
                    "For --llm-type 'openai-v1', you must provide --llm-ip and "
                    "--llm-model."
                )
            generated_text = _generate_docstring_openai_v1(
                code_snippet, element_type, args.llm_model, args.llm_ip, args.api_key
            )

        # Common cleanup
        if generated_text.strip().startswith("```python"):
            generated_text = generated_text.strip()[9:].strip()
        if generated_text.strip().startswith("```"):
            generated_text = generated_text.strip()[3:].strip()
        if generated_text.strip().endswith("```"):
            generated_text = generated_text.strip()[:-3].strip()

        logging.info("<- Received docstring from LLM.")
        return generated_text

    except requests.exceptions.RequestException as e:
        logging.error(f"{Colors.RED}Error calling LLM API: {e}{Colors.ENDC}")
        return f"/* Error generating docstring: {e} */"
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logging.error(f"{Colors.RED}Error parsing LLM response: {e}{Colors.ENDC}")
        return f"/* Error parsing LLM response {e}*/"
    except ValueError as e:
        logging.error(f"{Colors.RED}Configuration Error: {e}{Colors.ENDC}")
        return f"/* Configuration error: {e} */"


def parse_flake8_input(input_stream):
    """Parses flake8 output from a stream for missing docstring errors (D100-D103)."""
    # Example: ./sample_module.py:27:5: D102 Missing docstring in public method
    flake8_pattern = re.compile(
        r"^(?P<file_path>[^:]+):(?P<line>\d+):(?P<col>\d+): (?P<error>D10[0-7])"
    )
    for line in input_stream:
        match = flake8_pattern.match(line)
        if match:
            data = match.groupdict()
            data["line"] = int(data["line"])
            yield data


def find_element_at_line(file_path, line_number):
    """Finds a class or function definition at a specific line in a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()

        tree = ast.parse(source_code)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.lineno == line_number:
                    return {"node": node, "source_code": source_code}
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")

    return None


def add_docstring_to_file(file_path, node, docstring):
    """Inserts a docstring into a Python file for a given AST node."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            original_lines = f.readlines()

        if docstring.startswith('"""'):
            docstring = docstring.strip()[3:].strip()
        if docstring.endswith('"""'):
            docstring = docstring.strip()[:-3].strip()

        if not node.body:
            logging.warning(f"  - Cannot add docstring to empty body in {node.name}.")
            return False

        docstring_indent_level = node.body[0].col_offset
        docstring_indent = " " * docstring_indent_level

        docstring_lines = docstring.strip().split("\n")

        if len(docstring_lines) == 1:
            formatted_docstring_content = (
                f'{docstring_indent}"""{docstring_lines[0]}"""\n'
            )
        else:
            formatted_lines = [f'{docstring_indent}"""{docstring_lines[0]}']
            formatted_lines.extend(
                [f"{docstring_indent}{line}" for line in docstring_lines[1:]]
            )
            formatted_lines.append(f'{docstring_indent}"""')
            formatted_docstring_content = "\n".join(formatted_lines) + "\n"

        insertion_line_index = node.body[0].lineno - 1
        original_lines.insert(insertion_line_index, formatted_docstring_content)

        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(original_lines)

        logging.info(f"  - Successfully updated file: {file_path}")
        return True

    except Exception as e:
        logging.error(f"  - Error updating file {file_path}: {e}")
        return False


def add_module_docstring_to_file(file_path, docstring):
    """Inserts a module-level docstring at the top of a Python file."""

    if docstring.startswith('"""'):
        docstring = docstring.strip()[3:].strip()
    if docstring.endswith('"""'):
        docstring = docstring.strip()[:-3].strip()

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            original_lines = f.readlines()

        # Format the docstring for file insertion
        docstring_lines = docstring.strip().split("\n")
        if len(docstring_lines) == 1:
            # Add extra newline for separation
            formatted_docstring = f'"""{docstring_lines[0]}"""\n\n'
        else:
            formatted_lines = [f'"""{docstring_lines[0]}']
            formatted_lines.extend(docstring_lines[1:])
            formatted_lines.append('"""')
            # Add extra newline for separation
            formatted_docstring = "\n".join(formatted_lines) + "\n\n"

        # Find where to insert the docstring (after any shebang or encoding
        # declarations)
        insertion_index = 0
        for i, line in enumerate(original_lines):
            stripped_line = line.strip()
            if stripped_line.startswith("#!/") or (
                stripped_line.startswith("#") and "coding" in stripped_line
            ):
                insertion_index = i + 1
            elif stripped_line:
                # Stop at the first line of actual code or import
                break

        # Insert the formatted docstring
        original_lines.insert(insertion_index, formatted_docstring)

        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(original_lines)

        logging.info(f"  - Successfully added module docstring to: {file_path}")
        return True

    except Exception as e:
        logging.error(f"  - Error updating module docstring for {file_path}: {e}")
        return False


def main(args):
    """Main processing loop to read from stdin and patch files."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logging.info("Reading flake8 output from stdin...")
    flake8_errors = list(parse_flake8_input(sys.stdin))

    if not flake8_errors:
        logging.info("No missing docstring errors (D100-D107) found in flake8 output.")
        return

    logging.info(f"Found {len(flake8_errors)} missing docstring errors. Processing...")

    for error in flake8_errors:
        file_path = error["file_path"]
        line_num = error["line"]
        logging.info(
            "\nProcessing error "
            f"{error['error']} in {file_path} at line {line_num}"
        )

        generated_docstring = None
        node_for_update = None  # Stores the AST node for function/class updates

        if error["error"] in ["D100", "D104"]:
            # Handle missing module/package docstring
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    source_code = f.read()

                if not source_code.strip() and file_path.endswith("__init__.py"):
                    logging.warning(f"  - Skipping empty __init__.py file: {file_path}")
                    continue

                # Safeguard: check if a docstring already exists.
                if ast.get_docstring(ast.parse(source_code)):
                    logging.warning(
                        "  - Module/Package already has a docstring. Skipping."
                    )
                    continue

                element_type = "Package" if error["error"] == "D104" else "Module"
                generated_docstring = generate_docstring_with_llm(
                    source_code, element_type, args
                )
            except Exception as e:
                logging.error(
                    "  - An error occurred while processing "
                    f"module/package {file_path}: {e}"
                )
                continue
        else:
            # Handle missing class/function/method docstring (D101, D102, D103, D105,
            # D106, D107)
            element_info = find_element_at_line(file_path, line_num)

            if not element_info:
                logging.warning(
                    f"  - Could not find a class/function at line {line_num}. Skipping."
                )
                continue

            node_for_update = element_info["node"]
            source_code = element_info["source_code"]

            element_code = get_code_snippet(source_code, node_for_update)
            element_type = (
                "Class" if isinstance(node_for_update, ast.ClassDef) else "Function"
            )
            generated_docstring = generate_docstring_with_llm(
                element_code, element_type, args
            )

        if (
            not generated_docstring
            or "Error" in generated_docstring
            or "Skipped" in generated_docstring
        ):
            logging.warning(
                f"{Colors.YELLOW}  - Failed to generate a valid docstring for "
                f"{file_path}. Skipping update.{Colors.ENDC}"
            )
            print(f"Failed docstring at {file_path}:{line_num}:")
            print(generated_docstring)
            continue

        print("-" * 20 + " Generated Docstring " + "-" * 15)
        print('"""')
        print(generated_docstring)
        print('"""')

        if args.apply_changes:
            logging.info(f"-> Applying changes to {file_path}...")
            if error["error"] in ["D100", "D104"]:
                add_module_docstring_to_file(file_path, generated_docstring)
            elif node_for_update:
                add_docstring_to_file(file_path, node_for_update, generated_docstring)
            else:
                logging.error(
                    f"  - Could not apply changes for {file_path} due to "
                    "missing element info."
                )
        else:
            logging.info(
                "-> In dry-run mode. To apply changes, run with the "
                "--apply-changes flag."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Reads flake8 docstring errors from stdin, generates docstrings "
            "using a selected LLM, "
            "and inserts them into the source code."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--apply-changes",
        action="store_true",
        help=(
            "Actually modify the files with the generated docstrings. "
            "Default is a dry-run."
        ),
    )
    # LLM Configuration Arguments
    parser.add_argument(
        "--llm-type",
        type=str,
        choices=["gemini", "ollama", "openai-v1"],
        default="gemini",
        help="The type of LLM endpoint to use. Default: gemini.",
    )
    parser.add_argument(
        "--llm-ip",
        type=str,
        default="http://localhost:11434",
        help=(
            "The IP address and port for 'ollama' or 'openai-v1' types "
            "(e.g., http://localhost:8000)."
        ),
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        help=(
            "The specific model name to use for 'ollama' or 'openai-v1' "
            "types (e.g., 'llama3')."
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_API_KEY,
        help=(
            "API key for the LLM service. Can also be set via MY_LLM_API_KEY "
            "environment variable."
        ),
    )

    args = parser.parse_args()

    main(args)
