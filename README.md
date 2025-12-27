# Canto DSL

A **neurosymbolic language** that abstracts prompt engineering into declarative predicates.

Canto combines LLM semantic understanding with **Defeasible Logic Programming (DeLP)** - an argumentation-based reasoning framework. You define predicates and semantic categories; Canto handles the prompting, output parsing, and resolves conflicts through formal argumentation when multiple predicates could apply.

## Why Canto?

**Traditional prompt engineering:**
```
"Extract the food item from the text. If it mentions drying methods like
'dried', 'dehydrated', or 'sun-dried', mark it as dried. But if it's also
smoked, prioritize that. Unless it's a ready meal, then ignore processing..."
```

**With Canto:**
```canto
?drying_terms resembles "dried", "dehydrated", "sun-dried"
?smoking_terms resembles "smoked", "smoke-cured"

?form becomes "dried" when ?input is like ?drying_terms
?form becomes "smoked" when ?input is like ?smoking_terms overriding all
```

Canto gives you:
- **Neurosymbolic reasoning**: LLM semantic matching + DeLP argumentation
- **Declarative predicates**: Define what to extract, not how to prompt
- **Semantic categories**: Abstract fuzzy matching into named concepts
- **Defeasible logic**: Handle exceptions and priorities through argumentation
- **Conflict detection**: Catch predicate ambiguities at build time

## Installation

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/your-org/canto-lang.git
cd canto-lang
uv sync
```

## Quick Start

### 1. Create a `.canto` file

```canto
"""
Classify user input as either fruit or vehicle.
"""

// Define semantic categories - the LLM matches similar terms
?fruit_terms resembles "apple", "orange", "banana", "grape"
?vehicle_terms resembles "car", "truck", "bike", "bus"

// Define variables
?input meaning "the user input"
?result can be "fruit", "vehicle"

// Predicates for classification
?result becomes "fruit" when ?input is like ?fruit_terms
?result becomes "vehicle" when ?input is like ?vehicle_terms
```

### 2. Build

```bash
uv run canto build your_file.canto
```

## CLI Usage

```bash
# Basic build
uv run canto build <file.canto>

# Verbose output
uv run canto build <file.canto> -v

# With optimization (requires evaluation data)
uv run canto build <file.canto> --optimize --eval-data examples.json
```

## Language Syntax

### Variables and Schemas

```canto
// Simple variable
?input meaning "the user's query text"

// Variable with allowed values
?category can be "A", "B", "C"

// Nested structure for complex extraction
?order meaning "a customer order" with
    ?items has a list of ?item meaning "ordered items"
    ?total meaning "order total amount"

?item meaning "a single item" with
    ?name meaning "item name"
    ?quantity meaning "quantity ordered"
```

### Semantic Categories

Define patterns that represent a concept. The LLM matches semantically similar terms:

```canto
?urgent_terms resembles "ASAP", "urgent", "emergency", "critical"
?polite_terms resembles "please", "thank you", "appreciate"
```

### Predicates

```canto
// Basic predicate
?priority becomes "high" when ?input is like ?urgent_terms

// Defeasible predicate (can be overridden)
normally ?priority becomes "normal"

// Strict predicate with override
?priority becomes "critical"
    when ?input is like ?emergency_terms
    overriding all

// Conjunctions
?status becomes "vip"
    when ?spend is like ?high_spend_terms
    and ?tenure is like ?long_tenure_terms

// Disjunctions
?status becomes "priority"
    when ?input is like ?urgent_terms
    or ?input is like ?important_terms
```

### Collection Predicates

```canto
// Existential quantifier
?has_fruit becomes true when ?items has any that is like ?fruit_terms

// Universal quantifier
?all_organic becomes true when ?items has all that is like ?organic_terms

// Negated existential
?meat_free becomes true when ?items has none that is like ?meat_terms

// Cardinality constraints
?bulk_order becomes true when length of ?items where is like ?bulk_terms > 5
```

## Python Concepts

Canto's `Concept` API is grounded in **Prototype Theory** (Eleanor Rosch, 1973) — the idea that humans categorize things not through rigid definitions, but through **prototypical exemplars**. We don't define "bird" with necessary and sufficient conditions; we recognize birds by their similarity to robins and sparrows.

Classical categorization (from Aristotle onward) assumes categories have sharp boundaries: something either *is* or *isn't* a member based on a checklist of features. But Rosch's experiments showed that categories are actually **graded** — a robin is a "better" bird than a penguin, and we judge membership by *family resemblance* to central examples. This explains why traditional rule-based systems struggle with natural language: they demand crisp boundaries where human cognition sees gradients. LLMs, trained on human text, inherit this prototype-based reasoning. When you give an LLM exemplars like "chest pain" and "difficulty breathing," it doesn't match strings — it recognizes the *semantic neighborhood* those examples define. Canto's `resembles` clause makes this cognitive model explicit and composable.

### Defining Concepts

The `resembles` clause defines prototypes — the LLM judges membership by similarity:

```python
from canto_core import Concept

# Prototypical emergency symptoms
emergency_symptoms = (
    Concept("emergency_symptoms")
    .resembles("chest pain", "difficulty breathing", "severe bleeding")
    .meaning("symptoms requiring immediate attention")
)

# "Shortness of breath" matches because it's similar to the prototypes
```

Concepts can have nested structure with `has`:

```python
patient = (
    Concept("patient")
    .meaning("extracted patient information")
    .has(
        Concept("name").meaning("patient's full name"),
        Concept("age").meaning("patient's age in years"),
        Concept("chief_complaint").meaning("primary reason for visit"),
        Concept("symptoms").has(
            Concept("text").meaning("symptom description"),
            Concept("severity").can_be("mild", "moderate", "severe"),
        ),
    )
)
```

And constrained values with `can_be`:

```python
triage_level = (
    Concept("triage_level")
    .meaning("urgency classification")
    .can_be("immediate", "urgent", "delayed", "minor")
    .resembles("priority", "severity", "acuity")
)
```

### Building with Resolution

The `CantoBuilder` validates that all references in your `.canto` file resolve to declared concepts or variables:

```python
from canto_core import CantoBuilder, ResolutionErrors

builder = CantoBuilder()
builder.register_concept(patient)
builder.register_concept(emergency_symptoms)

try:
    result = builder.build("triage.canto")
except ResolutionErrors as e:
    print(e)  # "Unresolved reference: 'unknown_concept'"
```

The `BuildResult` contains the parsed AST, symbol table, and registered concepts:

```python
result = builder.build("triage.canto")

# Access the symbol table
print("patient" in result.symbols)  # True

symbol = result.symbols.resolve("patient")
print(symbol.kind)    # SymbolKind.CONCEPT
print(symbol.source)  # The Concept object

# Access registered concepts
print(result.concepts.keys())  # dict_keys(['patient', 'emergency_symptoms'])
```

### Generating Prompts

The `PromptGenerator` combines concepts with parsed rules to generate prompts:

```python
from canto_core import PromptGenerator

result = builder.build("triage.canto")
generator = PromptGenerator(result)
prompt = generator.generate()

print(prompt)
```

### Complete Example

**concepts.py**
```python
from canto_core import Concept

patient = (
    Concept("patient")
    .meaning("extracted patient information")
    .has(
        Concept("name").meaning("patient's full name"),
        Concept("chief_complaint").meaning("primary reason for visit"),
    )
)

emergency_symptoms = (
    Concept("emergency_symptoms")
    .resembles("chest pain", "difficulty breathing", "severe bleeding", "loss of consciousness")
    .meaning("symptoms requiring immediate attention")
)
```

**triage.canto**
```canto
"""
Medical triage classification.
"""

?is_emergency can be true, false meaning "whether case requires immediate attention"

?is_emergency becomes true
    when ?chief_complaint of ?patient is like ?emergency_symptoms
    overriding all

normally ?is_emergency becomes false
```

**build.py**
```python
from canto_core import CantoBuilder, PromptGenerator
from concepts import patient, emergency_symptoms

builder = CantoBuilder()
builder.register_concept(patient)
builder.register_concept(emergency_symptoms)

result = builder.build("triage.canto")

generator = PromptGenerator(result)
prompt = generator.generate()
print(prompt)
```

## Running Tests

```bash
uv run pytest
uv run pytest packages/core/tests/test_mutual_exclusivity.py -v
```

## Examples

See the `examples/` directory:
- `medical_triage_has.canto` - Medical triage classification
- `cleanlab_extractor.canto` - Data extraction pipeline
