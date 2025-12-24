"""
Prolog utility functions for term normalization and formatting.
"""

import unicodedata


def normalize_prolog_atom(term: str) -> str:
    """
    Normalize a term to a valid unquoted Prolog atom.

    - Converts unicode to ASCII
    - Lowercases everything
    - Removes apostrophes and quotes
    - Replaces spaces and hyphens with underscores
    - Keeps only alphanumeric and underscore characters
    - Ensures the atom starts with a letter

    Examples:
        "sott'olio" -> "sottolio"
        "sun-dried" -> "sun_dried"
        "café" -> "cafe"
        "don't" -> "dont"
    """
    # Normalize unicode to ASCII
    normalized = unicodedata.normalize('NFKD', term)
    normalized = normalized.encode('ascii', 'ignore').decode('ascii')

    # Lowercase
    normalized = normalized.lower()

    # Remove apostrophes and quotes
    normalized = normalized.replace("'", "").replace('"', "")

    # Replace spaces and hyphens with underscores
    normalized = normalized.replace(" ", "_").replace("-", "_")

    # Keep only alphanumeric and underscore
    normalized = ''.join(c if c.isalnum() or c == '_' else '' for c in normalized)

    # Clean up underscores
    while '__' in normalized:
        normalized = normalized.replace('__', '_')
    normalized = normalized.strip('_')

    # Ensure starts with letter
    if normalized and not normalized[0].isalpha():
        normalized = 'x_' + normalized

    return normalized or 'unknown'
