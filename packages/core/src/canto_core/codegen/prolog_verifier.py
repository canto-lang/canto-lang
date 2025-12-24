"""
Prolog Syntax Verifier

Validates LLM-generated Prolog code by parsing it via Prolog helper predicates.
"""

from typing import Tuple, Optional
from pathlib import Path
from janus_swi import query_once, consult


class PrologSyntaxVerifier:
    """
    Verifies Prolog code syntax by parsing it with SWI-Prolog via Janus
    """

    _helpers_loaded = False  # Class variable to track if helpers are loaded

    def __init__(self):
        """Initialize and load helper predicates into Prolog"""
        self._ensure_helpers_loaded()

    def _ensure_helpers_loaded(self):
        """Load Prolog helper file if not already loaded"""
        if PrologSyntaxVerifier._helpers_loaded:
            return  # Already loaded

        # Load the helper file
        helpers_file = Path(__file__).parent / "prolog_helpers.pl"

        if not helpers_file.exists():
            raise FileNotFoundError(f"Helper file not found: {helpers_file}")

        try:
            consult(str(helpers_file))
            PrologSyntaxVerifier._helpers_loaded = True
            # print(f"✓ Loaded Prolog helpers from {helpers_file}")
        except Exception as e:
            # Might already be loaded, check if predicate exists
            try:
                query_once("current_predicate(verify_prolog_syntax/2)")
                PrologSyntaxVerifier._helpers_loaded = True
                # Already loaded, no problem
            except:
                # Really failed
                raise RuntimeError(f"Failed to load Prolog helpers: {e}")

    def verify_syntax(self, prolog_code: str) -> Tuple[bool, Optional[str]]:
        """
        Verify that Prolog code is syntactically valid

        Args:
            prolog_code: Prolog code to verify

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if code is syntactically correct
            - error_message: None if valid, error description if invalid
        """
        if not prolog_code or not prolog_code.strip():
            return False, "Empty Prolog code"

        try:
            # Call the Prolog helper predicate
            # Stream handling is entirely within Prolog, Janus only sees simple values
            result = query_once(f"verify_prolog_syntax({repr(prolog_code)}, Result)")

            if not result:
                return False, "No result from verification"

            result_value = result.get('Result')

            # Check if result is the atom 'true' (success)
            # Janus may return it as boolean True or string 'true'
            if result_value == True or result_value == 'true':
                return True, None

            # Otherwise it's an error term error(...)
            # Note: Complex error terms with streams can't serialize,
            # so we catch those in the except block
            return False, f"Prolog syntax error: {result_value}"

        except Exception as e:
            # Catches serialization errors from complex error terms
            error_msg = str(e)
            # Extract just the syntax error part if present
            if "syntax_error" in error_msg:
                import re
                match = re.search(r"syntax_error\(([^)]+)\)", error_msg)
                if match:
                    return False, f"Syntax error: {match.group(1)}"
            return False, f"Verification error: {error_msg}"

    def verify_and_explain(self, prolog_code: str) -> str:
        """
        Verify Prolog code and return human-readable result

        Args:
            prolog_code: Prolog code to verify

        Returns:
            Human-readable verification result
        """
        is_valid, error_msg = self.verify_syntax(prolog_code)

        if is_valid:
            return "✓ Prolog syntax is valid"
        else:
            return f"✗ Syntax error: {error_msg}"


def verify_prolog_code(prolog_code: str) -> bool:
    """
    Convenience function to verify Prolog code

    Args:
        prolog_code: Prolog code to verify

    Returns:
        True if valid, False otherwise
    """
    verifier = PrologSyntaxVerifier()
    is_valid, _ = verifier.verify_syntax(prolog_code)
    return is_valid
