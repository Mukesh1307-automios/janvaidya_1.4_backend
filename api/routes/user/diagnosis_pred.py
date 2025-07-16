# Add these imports to your existing file
from fastapi import HTTPException, APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import logging
from datetime import datetime
import uuid
from utils.helper import fix_common_json_errors, determine_age_group, logger

# Database imports
from sqlalchemy import text
from db.database import get_db

# LLM imports
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate

# ================================
# NEW MODELS FOR DIAGNOSIS
# ================================

class Option(BaseModel):
    id: int
    text: str
    childQuestion: Optional[int] = None
    colorIndex: Optional[int] = None  # Indicates if this option was selected

class QuestionNode(BaseModel):
    id: int
    question: str
    age_group: str
    gender: str
    options: Dict[str, Option]

class QuestionnaireData(BaseModel):
    rootQuestion: int
    nodes: Dict[str, QuestionNode]

class DiagnosisRequest(BaseModel):
    protocol_id: int
    doctor_id: int
    user_responses: List[QuestionnaireData]  # Changed to accept questionnaire structure

class DiagnosisDetail(BaseModel):
    diagnosis_title: str
    description: str

class RedFlagSymptom(BaseModel):
    symptom_no: int
    symptom_name: str
    description: str
    urgency_level: str  # High/Medium/Low

class OTCMedication(BaseModel):
    otc_no: int
    title: str
    name: str
    dosage_duration: str
    type: str
    intake_type: str
    intake_schedules: str

class AdviceItem(BaseModel):
    advice_no: int
    title: str
    description: str

class PrecautionItem(BaseModel):
    precaution_no: int
    title: str
    description: str
    importance: str  # Critical/Important/General

class LabTest(BaseModel):
    test_no: int
    test_name: str
    purpose: str
    urgency: str  # Immediate/Within 24h/Within week/Routine
    normal_range: Optional[str] = None

class DietRecommendation(BaseModel):
    diet_no: int
    category: str  # Foods to Include/Foods to Avoid/Meal Timing/Hydration
    title: str
    description: str
    duration: str

class DiagnosisResponse(BaseModel):
    q_id: int
    q_tag: str = "DIAGNOSIS"
    reason: str  # Why this diagnosis was given based on symptoms
    diagnosis: DiagnosisDetail
    red_flag_symptom: List[RedFlagSymptom]
    over_the_counter: List[OTCMedication]
    advice: List[AdviceItem]
    precaution: List[PrecautionItem]
    lab_test: List[LabTest]
    diet: List[DietRecommendation]

# ================================
# COMPREHENSIVE MEDICAL PROMPT FOR MEDGEMMA
# ================================

COMPREHENSIVE_MEDICAL_PROMPT = """
You are an expert medical AI assistant. Analyze the patient information and provide a complete medical assessment covering all aspects of care.

PATIENT INFORMATION:
{patient_summary}

INSTRUCTIONS:
Based on the patient's symptoms, demographics, and presentation, provide a comprehensive medical evaluation covering:
1. Primary diagnosis with concise explanation
2. Clear reasoning for why this diagnosis was chosen
3. Red flag symptoms to monitor (only if relevant)
4. Over-the-counter medications (only if needed)
5. Essential medical advice (1-2 most important items)
6. Critical precautions (only if necessary)
7. Laboratory tests (only if clinically indicated)
8. Dietary recommendations (only essential ones)

RESPONSE FORMAT - You must respond with a valid JSON object:

{{
    "reason": "Clear explanation of why this diagnosis was made based on the patient's specific symptoms and presentation",
    "diagnosis": {{
        "diagnosis_title": "Primary medical diagnosis name",
        "description": "Concise medical explanation (2-3 sentences max) focusing on the key aspects of this condition relevant to the patient"
    }},
    "red_flag_symptom": [
        {{
            "symptom_no": 1,
            "symptom_name": "Critical warning symptom",
            "description": "When this symptom indicates serious condition requiring immediate attention",
            "urgency_level": "High"
        }}
    ],
    "over_the_counter": [
        {{
            "otc_no": 1,
            "title": "Primary symptom relief",
            "name": "Specific OTC medication name",
            "dosage_duration": "X Days",
            "type": "Tablet/Syrup/Ointment",
            "intake_type": "Clear administration instructions with timing and dosage",
            "intake_schedules": "101"
        }}
    ],
    "advice": [
        {{
            "advice_no": 1,
            "title": "Most Important Care",
            "description": "The single most important care instruction for recovery"
        }},
        {{
            "advice_no": 2,
            "title": "Key Recovery Step",
            "description": "Second most critical advice for patient recovery"
        }}
    ],
    "precaution": [
        {{
            "precaution_no": 1,
            "title": "Critical Safety",
            "description": "Most important safety measure or restriction",
            "importance": "Critical"
        }}
    ],
    "lab_test": [
        {{
            "test_no": 1,
            "test_name": "Essential diagnostic test",
            "purpose": "Why this specific test is needed",
            "urgency": "Immediate/Within 24h/Within week/Routine",
            "normal_range": "Normal values (if applicable)"
        }}
    ],
    "diet": [
        {{
            "diet_no": 1,
            "category": "Foods to Include",
            "title": "Most beneficial foods",
            "description": "Specific foods that will most help recovery",
            "duration": "During recovery"
        }},
        {{
            "diet_no": 2,
            "category": "Foods to Avoid",
            "title": "Foods that worsen condition",
            "description": "Key foods to avoid and why",
            "duration": "Until symptoms resolve"
        }}
    ]
}}

QUALITY REQUIREMENTS:
- Provide only the MOST ESSENTIAL items in each category (1-2 items preferred, 3-4 only if absolutely necessary)
- Make descriptions concise but actionable
- Focus on patient-specific relevance
- Include only clinically necessary recommendations
- Prioritize practical, immediate value to the patient
- Ensure every recommendation directly relates to the patient's symptoms

Generate your focused medical assessment now:
"""

# ================================
# HELPER FUNCTIONS
# ================================

def create_patient_summary(questionnaire_data_list: List[QuestionnaireData]) -> str:
    """Create a comprehensive medical summary from questionnaire data with selected options."""
    
    demographics = []
    symptoms = []
    timeline = []
    severity_indicators = []
    associated_symptoms = []
    
    # Process each questionnaire in the list (typically just one)
    for questionnaire_data in questionnaire_data_list:
        nodes = questionnaire_data.nodes
        
        # Extract selected responses from each node
        for node_key, node in nodes.items():
            question_text = node.question
            
            # Find selected options (those with colorIndex)
            for option_key, option in node.options.items():
                if option.colorIndex is not None:  # This option was selected
                    text = option.text.lower()
                    
                    # Extract demographics
                    if any(gender in text for gender in ['male', 'female']):
                        demographics.append(f"{option.text} patient")
                    elif any(age_term in text for age_term in ['years', 'age', 'old']):
                        demographics.append(f"aged {option.text}")
                        
                    # Extract temporal information
                    elif any(time_term in text for time_term in ['week', 'day', 'month', 'hour', 'minute']):
                        timeline.append(f"duration: {option.text}")
                        
                    # Extract severity indicators
                    elif any(severity in text for severity in ['severe', 'mild', 'moderate', 'intense', 'sharp', 'dull']):
                        severity_indicators.append(f"severity: {option.text}")
                        
                    # Extract primary symptoms
                    elif any(primary in text for primary in ['cough', 'pain', 'fever', 'headache', 'nausea', 'vomiting']):
                        symptoms.append(option.text)
                        
                    # Extract associated symptoms  
                    elif any(assoc in text for assoc in ['chills', 'fatigue', 'weakness', 'shortness', 'breathing', 'chest']):
                        associated_symptoms.append(option.text)
                        
                    # Catch-all for other responses
                    else:
                        symptoms.append(option.text)
    
    # Build comprehensive summary
    summary_parts = []
    
    # Demographics
    if demographics:
        summary_parts.append(" ".join(demographics))
    
    # Primary presentation
    if symptoms:
        if len(symptoms) == 1:
            summary_parts.append(f"presenting with {symptoms[0]}")
        else:
            summary_parts.append(f"presenting with {', '.join(symptoms[:-1])} and {symptoms[-1]}")
    
    # Timeline
    if timeline:
        summary_parts.append(f"({', '.join(timeline)})")
    
    # Severity
    if severity_indicators:
        summary_parts.append(f"characterized by {', '.join(severity_indicators)}")
    
    # Associated symptoms
    if associated_symptoms:
        summary_parts.append(f"with associated {', '.join(associated_symptoms)}")
    
    return " ".join(summary_parts).strip()

async def generate_comprehensive_diagnosis_with_llm(patient_summary: str) -> Dict[str, Any]:
    """Generate comprehensive diagnosis using MedGemma model."""
    try:
        # Initialize MedGemma model with optimal settings
        llm = ChatOllama(
            model="symptoma/medgemma3:27b",
            base_url="http://localhost:11434",
            temperature=0.1,  # Very low for consistent medical responses
            top_p=0.95,
            repeat_penalty=1.05,
            num_ctx=8192,  # Extended context for comprehensive medical reasoning
        )

        # Create comprehensive diagnosis prompt
        prompt_template = PromptTemplate(
            template=COMPREHENSIVE_MEDICAL_PROMPT,
            input_variables=["patient_summary"]
        )

        formatted_prompt = prompt_template.format(patient_summary=patient_summary)

        logger.info(f"Generating comprehensive diagnosis for: {patient_summary}")

        # Generate comprehensive diagnosis
        response = llm.invoke(formatted_prompt)
        content = response.content.strip()
        
        # Clean JSON response
        content = clean_json_response(content)
        diagnosis_data = json.loads(content)
        
        logger.info(f"Successfully generated comprehensive diagnosis: {diagnosis_data['diagnosis']['diagnosis_title']}")
        return diagnosis_data

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in comprehensive diagnosis: {e}")
        logger.error(f"Raw content: {content}")
        raise HTTPException(status_code=500, detail="Failed to parse comprehensive medical diagnosis response")
    
    except Exception as e:
        logger.error(f"Error generating comprehensive diagnosis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate comprehensive diagnosis: {e}")

def clean_json_response(content: str) -> str:
    """Clean and prepare JSON response from LLM."""
    # Remove markdown formatting
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    
    # Fix common JSON errors
    content = fix_common_json_errors(content.strip())
    
    return content

# ================================
# COMPREHENSIVE DIAGNOSIS API ENDPOINT
# ================================


router = APIRouter(prefix='/GET', tags=['Diagnosis Prediction'])

@router.post("/GenerateDiagnosis", response_model=DiagnosisResponse)
async def generate_diagnosis(request: DiagnosisRequest):
    """
    Generate comprehensive medical diagnosis based on user's questionnaire responses.
    
    Uses MedGemma AI to analyze symptoms and provide complete medical assessment including:
    - Primary diagnosis with detailed medical explanation
    - Red flag symptoms to monitor for complications
    - OTC medication recommendations with detailed dosing
    - Structured medical advice and care guidelines
    - Important precautions and safety measures
    - Laboratory tests for diagnosis confirmation
    - Comprehensive dietary recommendations
    """
    try:
        # Validate protocol and doctor exist
        with get_db() as conn:
            protocol_exists = conn.execute(
                text("SELECT COUNT(*) FROM medapp.protocols WHERE id = :id"), 
                {"id": request.protocol_id}
            ).scalar()
            
            doctor_exists = conn.execute(
                text("SELECT COUNT(*) FROM medapp.doctors WHERE doc_id = :doc_id"), 
                {"doc_id": request.doctor_id}
            ).scalar()
            
            if protocol_exists == 0:
                raise HTTPException(status_code=404, detail=f"Protocol ID {request.protocol_id} not found")
            if doctor_exists == 0:
                raise HTTPException(status_code=404, detail=f"Doctor ID {request.doctor_id} not found")

        # Validate user responses
        if not request.user_responses:
            raise HTTPException(status_code=400, detail="No questionnaire data provided")

        # Create comprehensive patient summary from questionnaire data
        patient_summary = create_patient_summary(request.user_responses)
        
        logger.info(f"Processing comprehensive diagnosis request for questionnaire data")
        logger.info(f"Patient summary: {patient_summary}")
        
        # Generate comprehensive diagnosis using MedGemma
        diagnosis_result = await generate_comprehensive_diagnosis_with_llm(patient_summary)
        
        # Generate q_id (no database storage)
        diagnosis_q_id = 1001  # Static or you can use uuid/timestamp
        
        # Extract diagnosis reasoning and title
        diagnosis_reason = diagnosis_result.get('reason', 'Diagnosis based on clinical assessment of presented symptoms')
        diagnosis_title = diagnosis_result['diagnosis']['diagnosis_title']

        logger.info(f"Generated comprehensive diagnosis: {diagnosis_title}")
        logger.info(f"Components generated - Red flags: {len(diagnosis_result.get('red_flag_symptom', []))}, OTC: {len(diagnosis_result.get('over_the_counter', []))}, Advice: {len(diagnosis_result.get('advice', []))}, Precautions: {len(diagnosis_result.get('precaution', []))}, Lab tests: {len(diagnosis_result.get('lab_test', []))}, Diet: {len(diagnosis_result.get('diet', []))}")
        
        # Create comprehensive structured response
        response = DiagnosisResponse(
            q_id=diagnosis_q_id,
            q_tag="DIAGNOSIS",
            reason=diagnosis_reason,
            diagnosis=DiagnosisDetail(
                diagnosis_title=diagnosis_result['diagnosis']['diagnosis_title'],
                description=diagnosis_result['diagnosis']['description']
            ),
            red_flag_symptom=[
                RedFlagSymptom(**symptom) for symptom in diagnosis_result.get('red_flag_symptom', [])
            ],
            over_the_counter=[
                OTCMedication(**med) for med in diagnosis_result.get('over_the_counter', [])
            ],
            advice=[
                AdviceItem(**adv) for adv in diagnosis_result.get('advice', [])
            ],
            precaution=[
                PrecautionItem(**prec) for prec in diagnosis_result.get('precaution', [])
            ],
            lab_test=[
                LabTest(**test) for test in diagnosis_result.get('lab_test', [])
            ],
            diet=[
                DietRecommendation(**diet_rec) for diet_rec in diagnosis_result.get('diet', [])
            ]
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_diagnosis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ================================
# OPTIONAL: SYMPTOM ANALYSIS ENDPOINT  
# ================================

@router.post("/AnalyzeSymptoms")
async def analyze_symptoms(request: DiagnosisRequest):
    """
    Analyze patient symptoms from questionnaire data for preliminary assessment.
    """
    try:
        patient_summary = create_patient_summary(request.user_responses)
        
        # Extract selected options for detailed view
        selected_responses = []
        for questionnaire_data in request.user_responses:
            for node_key, node in questionnaire_data.nodes.items():
                for option_key, option in node.options.items():
                    if option.colorIndex is not None:
                        selected_responses.append({
                            "question_id": node.id,
                            "question": node.question,
                            "selected_option_id": option.id,
                            "selected_text": option.text,
                            "color_index": option.colorIndex
                        })
        
        return JSONResponse(content={
            "patient_summary": patient_summary,
            "total_selected_responses": len(selected_responses),
            "selected_responses": selected_responses
        })
        
    except Exception as e:
        logger.error(f"Error analyzing symptoms: {e}")
        raise HTTPException(status_code=500, detail=str(e))