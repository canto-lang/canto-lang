"""
DeLP (Defeasible Logic Programming) - Reasoning Engine
"""

try:
    from .engine import JanusDeLP, create_janus_engine
    from .analyzer import DeLPReasoningAnalyzer, ReasoningPattern
    from .program import DeLPProgram
    from .models import DeLPRule, DeLPRuleSource, DeLPDeclaration, DeLPQueryResult
    __all__ = [
        'JanusDeLP', 'create_janus_engine',
        'DeLPReasoningAnalyzer', 'ReasoningPattern',
        'DeLPProgram', 'DeLPRule', 'DeLPRuleSource', 'DeLPDeclaration', 'DeLPQueryResult'
    ]
except ImportError as e:
    # Janus not available - but print why for debugging
    import sys
    print(f"[delp/__init__.py] Import failed: {e}", file=sys.stderr)
    __all__ = []
