"""Pydantic schemas for MedScribe AI clinical summary extraction."""

from __future__ import annotations

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field


class PatientDetails(BaseModel):
    name: Optional[str] = Field(default=None, description="Patient full name if mentioned")
    age: Optional[Union[int, str]] = Field(default=None, description="Patient age if mentioned (e.g. 45 or '45 years')")
    sex: Optional[str] = Field(default=None, description="Patient biological sex or gender if mentioned")
    identifiers: Optional[str] = Field(default=None, description="MRN, ID, or other patient identifiers if mentioned")


class HistoryOfPresentIllness(BaseModel):
    onset: Optional[str] = Field(default=None, description="When the current symptoms or condition began")
    duration: Optional[str] = Field(default=None, description="How long symptoms have persisted")
    progression: Optional[str] = Field(default=None, description="Worsening, improving, fluctuating, or stable trajectory")
    aggravating_factors: Optional[str] = Field(default=None, description="Triggers, positions, or activities that worsen the symptoms")
    relieving_factors: Optional[str] = Field(default=None, description="Interventions or rest that alleviate symptoms")


class Symptoms(BaseModel):
    positive: List[str] = Field(
        default_factory=list,
        description="Symptoms explicitly reported or affirmed by the patient or clinician"
    )
    negative: List[str] = Field(
        default_factory=list,
        description="Symptoms explicitly inquired about and denied by the patient (e.g. denies fever, no chest pain)"
    )


class PastMedicalHistory(BaseModel):
    conditions: List[str] = Field(default_factory=list, description="Chronic or previous diagnosed medical illnesses")
    surgeries: List[str] = Field(default_factory=list, description="Past surgical procedures")
    hospitalisations: List[str] = Field(default_factory=list, description="Past hospital admissions")


class MedicationEntry(BaseModel):
    name: str = Field(..., description="Name of the medication")
    dosage: Optional[str] = Field(default=None, description="Dose and frequency if specified (e.g. 500mg twice daily)")
    adherence: Optional[str] = Field(default=None, description="Patient adherence reported (e.g. compliant, missed doses, irregular)")


class ClinicalObservations(BaseModel):
    vitals: Dict[str, str] = Field(
        default_factory=dict,
        description="Objective vital signs mentioned aloud (e.g. {'BP': '120/80 mmHg', 'Pulse': '74 bpm', 'Temp': '98.6 F'})"
    )
    examination_findings: List[str] = Field(
        default_factory=list,
        description="Physical examination observations voiced by the clinician"
    )


class Plan(BaseModel):
    investigations: List[str] = Field(default_factory=list, description="Diagnostic tests, lab orders, or imaging ordered")
    prescriptions: List[str] = Field(default_factory=list, description="Medications prescribed, adjusted, or continued")
    advice: List[str] = Field(default_factory=list, description="Lifestyle advice, patient education, or non-pharmacological instructions")
    follow_up: Optional[str] = Field(default=None, description="Scheduled follow-up timing or conditions for urgent return")


class DialogueTurn(BaseModel):
    speaker: str = Field(..., description="Role of the speaker: 'Doctor', 'Patient', or 'Unknown'")
    text: str = Field(..., description="Verbatim text spoken by this speaker in this turn")


class ClinicalSummary(BaseModel):
    clinical_summary: str = Field(
        ...,
        description="Concise 2-4 sentence narrative synthesis of the consultation encounter"
    )
    patient_details: PatientDetails = Field(default_factory=PatientDetails)
    chief_complaint: List[str] = Field(
        default_factory=list,
        description="Primary presenting complaint(s) in patient's words where practical"
    )
    history_of_present_illness: HistoryOfPresentIllness = Field(default_factory=HistoryOfPresentIllness)
    symptoms: Symptoms = Field(default_factory=Symptoms)
    past_medical_history: PastMedicalHistory = Field(default_factory=PastMedicalHistory)
    medications: List[MedicationEntry] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list, description="Known drug, food, or environmental allergies")
    clinical_observations: ClinicalObservations = Field(default_factory=ClinicalObservations)
    assessment: List[str] = Field(default_factory=list, description="Clinician's diagnostic impression, differential, or conclusions")
    plan: Plan = Field(default_factory=Plan)


class AnalyzeResponse(ClinicalSummary):
    """Clinical summary extraction response including segmented dialogue turns."""
    turns: List[DialogueTurn] = Field(
        default_factory=list,
        description="Segmented dialogue turns with verbatim text and speaker labels"
    )

