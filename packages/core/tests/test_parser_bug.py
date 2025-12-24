
from canto_core.parser.dsl_parser import parse_string
from canto_core.ast_nodes import Condition

def test_parens():
    text = """
    $x IS true WHEN ($y IS true AND $z IS true)
    """
    try:
        rules = parse_string(text)
        rule = rules[0]
        print("Parse successful")
        print(rule.conditions)
        
        # Check structure
        cond = rule.conditions[0]
        print(f"Type of condition: {type(cond)}")
        print(f"Condition: {cond}")
        
        if isinstance(cond, list):
            print("BUG: Condition is a list!")
    except Exception as e:
        print(f"Parse failed: {e}")

if __name__ == "__main__":
    test_parens()
