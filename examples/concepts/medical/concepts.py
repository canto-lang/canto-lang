"""
Medical Domain Concepts

Semantic structures for medical triage and patient extraction.
"""

from canto_core.concept import Concept


# Patient extraction structure
patient = (
    Concept("patient")
    .meaning("extracted patient information")
    .has(
        Concept("name").meaning("patient's full name"),
        Concept("age").meaning("patient's age in years"),
        Concept("chief_complaint").meaning("primary reason for visit"),
        Concept("symptoms")
        .meaning("list of reported symptoms")
        .has(
            Concept("text").meaning("symptom description"),
            Concept("severity").can_be("mild", "moderate", "severe"),
            Concept("is_emergency").can_be(True, False),
        ),
        Concept("medications")
        .meaning("current medications")
        .has(
            Concept("med_name").meaning("medication name"),
            Concept("dosage").meaning("current dosage"),
        ),
    )
)

# Triage level classification
triage_level = (
    Concept("triage_level")
    .meaning("urgency classification for medical cases")
    .can_be("immediate", "urgent", "delayed", "minor")
    .resembles("priority", "severity", "acuity")
)

# Emergency symptom patterns
emergency_symptoms = (
    Concept("emergency_symptoms")
    .resembles(
        "chest pain",
        "difficulty breathing",
        "severe bleeding",
        "loss of consciousness",
        "stroke symptoms",
    )
    .meaning("symptoms requiring immediate attention")
)

# Diagnosis structure
diagnosis = (
    Concept("diagnosis")
    .meaning("identified medical condition")
    .resembles("condition", "disease", "ailment")
    .can_be("confirmed", "suspected", "ruled_out")
)
