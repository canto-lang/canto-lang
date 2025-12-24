"""
Low-level Prolog engine wrapper using Janus-SWI.

This module provides a clean abstraction over janus_swi for:
- Loading Prolog files
- Executing queries
- Managing temporary files
- Normalizing Prolog values to Python types
"""

from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path
import tempfile

from janus_swi import query_once, query, consult

from .models import normalize_prolog_value


class PrologEngine:
    """
    Low-level wrapper around Janus-SWI Prolog engine.

    Handles file loading, query execution, and value normalization.
    This class is stateless with respect to DeLP - it just manages
    Prolog interaction.
    """

    def __init__(self):
        self._temp_files: List[str] = []

    def consult_file(self, path: Path, description: str = "Prolog file") -> None:
        """
        Load a Prolog file into the engine.

        Args:
            path: Path to the Prolog file
            description: Human-readable description for error messages

        Raises:
            FileNotFoundError: If the file doesn't exist
            RuntimeError: If loading fails
        """
        if not path.exists():
            raise FileNotFoundError(f"{description} not found at {path}")

        try:
            consult(str(path))
            print(f"✓ Loaded {description} from {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {description}: {e}")

    def consult_string(self, prolog_code: str, description: str = "Prolog code") -> None:
        """
        Load Prolog code from a string.

        Creates a temporary file and consults it.

        Args:
            prolog_code: Prolog source code
            description: Human-readable description for logging
        """
        if not prolog_code.strip():
            return

        # Disable discontiguous warnings for dynamically generated code
        prolog_code = ":- style_check(-discontiguous).\n\n" + prolog_code

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False) as f:
                f.write(prolog_code)
                temp_path = f.name

            self._temp_files.append(temp_path)
            consult(temp_path)
            print(f"✓ Loaded {description} ({len(prolog_code)} bytes)")

        except Exception as e:
            raise RuntimeError(f"Failed to load {description}: {e}")

    def query_once(self, query_str: str, normalize: bool = True) -> Optional[Dict[str, Any]]:
        """
        Execute a Prolog query and return first result.

        Args:
            query_str: Prolog query string
            normalize: If True, normalize Prolog values to Python types

        Returns:
            Dict with variable bindings, or None if query fails
        """
        result = query_once(query_str)
        if normalize and result:
            return normalize_prolog_value(result)
        return result

    def query_all(self, query_str: str, normalize: bool = True) -> List[Dict[str, Any]]:
        """
        Execute a Prolog query and return all results.

        Args:
            query_str: Prolog query string
            normalize: If True, normalize Prolog values to Python types

        Returns:
            List of dicts with variable bindings
        """
        results = list(query(query_str))
        if normalize:
            return [normalize_prolog_value(r) for r in results]
        return results

    def query_iter(self, query_str: str, normalize: bool = True) -> Iterator[Dict[str, Any]]:
        """
        Execute a Prolog query and yield results lazily.

        Args:
            query_str: Prolog query string
            normalize: If True, normalize Prolog values to Python types

        Yields:
            Dicts with variable bindings
        """
        for result in query(query_str):
            if normalize:
                yield normalize_prolog_value(result)
            else:
                yield result

    def assert_fact(self, predicate: str, *args) -> None:
        """
        Add a fact to the knowledge base.

        Args:
            predicate: Predicate name
            args: Arguments to the predicate
        """
        args_str = ", ".join(str(arg) for arg in args)
        fact_str = f"{predicate}({args_str})"

        try:
            query_once(f"assertz({fact_str})")
            print(f"✓ Added fact: {fact_str}")
        except Exception as e:
            print(f"Failed to add fact: {e}")

    def retract_all(self, predicate: str, arity: int) -> None:
        """
        Remove all facts matching a predicate.

        Args:
            predicate: Predicate name
            arity: Number of arguments
        """
        pattern = ", ".join(["_"] * arity)
        try:
            query_once(f"retractall({predicate}({pattern}))")
        except Exception:
            pass  # Ignore if nothing to retract

    def abolish(self, predicate: str, arity: int) -> None:
        """
        Remove a predicate definition entirely.

        Args:
            predicate: Predicate name
            arity: Number of arguments
        """
        try:
            query_once(f"abolish({predicate}/{arity})")
        except Exception:
            pass  # Ignore errors

    def cleanup(self) -> None:
        """Clean up temporary files."""
        for temp_file in self._temp_files:
            try:
                Path(temp_file).unlink()
            except Exception:
                pass
        self._temp_files.clear()

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()
