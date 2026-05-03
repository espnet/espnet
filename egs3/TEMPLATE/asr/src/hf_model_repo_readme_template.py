"""Helpers for the default Hugging Face model repo README template."""

from pathlib import Path


def get_hf_model_repo_readme_template() -> str:
    """Return the default Hugging Face model repo README template.

    This helper loads the markdown template used by the publication stages
    when they generate ``README.md`` inside a packed model bundle.

    Returns:
        The template text from ``hf_model_repo_readme_template.md``.

    Raises:
        FileNotFoundError: If the template file is missing from the recipe
            source directory.
        OSError: If the template file cannot be read.

    Examples:
        >>> text = get_hf_model_repo_readme_template()
        >>> text.startswith("---")
        True
    """
    template_path = Path(__file__).parent / "hf_model_repo_readme_template.md"
    return template_path.read_text(encoding="utf-8")
