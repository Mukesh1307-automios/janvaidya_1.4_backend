from pathlib import Path
import logging
from fastapi import HTTPException

from typing import List, Dict, Any
import json
import logging
from models.schema import FilterRequest, Options ,MedicalConditionRequest,QuestionSetResponse, QuestionResponse
from db.database import get_db
# LangChain imports
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from sqlalchemy import text

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import prompts (you'll need to create this file)
from api.routes.user.AI_qna.prompt import QUESTIONS_ONLY_PROMPT



def get_file_type(filename: str) -> str:
    """Determine file type from filename"""
    extension = Path(filename).suffix.lower()
    
    if extension == '.pdf':
        return 'pdf'
    elif extension == '.txt':
        return 'txt'
    elif extension in ['.doc', '.docx']:
        return 'document'
    elif extension in ['.csv', '.xlsx', '.xls']:
        return 'data'
    else:
        return 'unsupported'
    

# ================================
# FULLY LLM GENERATION
# ================================

def fix_common_json_errors(text: str) -> str:
    """Fix common JSON formatting errors"""
    import re
    text = re.sub(r'"\s*\[(.*?)\]\s*"', lambda m: f'[{m.group(1)}]', text)
    text = re.sub(r'\[\s*"([^"\]]+)\]', r'["\1"]', text)
    text = re.sub(r',\s*([\]}])', r'\1', text)
    return text

def determine_age_group(age: int) -> str:
    """Determine age group based on age"""
    if 0 <= age <= 2:
        return "0-2"
    elif 3 <= age <= 12:
        return "3-12"
    elif 13 <= age <= 18:
        return "13-18"
    elif 19 <= age <= 40:
        return "19-40"
    elif 41 <= age <= 65:
        return "41-65"
    else:
        return "66+"

async def generate_questions_with_llm(protocol_id: int, doctor_id: int) -> Dict[str, Any]:
    """Generate questions using LLM based on protocol"""
    try:
        # Get protocol details from database
        with get_db() as conn:
            protocol_query = text("""
                SELECT name
                FROM medapp.protocols 
                WHERE id = :protocol_id
            """)
            
            protocol_result = conn.execute(protocol_query, {"protocol_id": protocol_id}).fetchone()
            
            if not protocol_result:
                raise HTTPException(status_code=404, detail=f"Protocol ID {protocol_id} not found")
            
            # Use protocol_name or description as medical_condition
            medical_condition = protocol_result.name  # or protocol_result.description
        
        llm = ChatOllama(
            model="symptoma/medgemma3:27b",
            base_url="http://localhost:11434",
            temperature=0.5
        )

        prompt_template = PromptTemplate(
            template=QUESTIONS_ONLY_PROMPT,
            input_variables=["medical_condition"]
        )

        formatted_prompt = prompt_template.format(medical_condition=medical_condition)
        
        logger.info(f"Generating questions for protocol {protocol_id}: {medical_condition}")
        response = llm.invoke(formatted_prompt)

        content = response.content.strip()
        
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        content = fix_common_json_errors(content.strip())
        questions_data = json.loads(content)
        
        logger.info(f"Successfully generated {len(questions_data.get('questions', []))} questions")
        return questions_data

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM response: {e}")
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate questions: {e}")

def filter_questions_by_demographics(questions: List[QuestionResponse], age: int = None, gender: str = None) -> List[QuestionResponse]:
    """Filter questions based on age and gender"""
    if age is None and gender is None:
        return questions
    
    filtered_questions = []
    target_age_group = determine_age_group(age) if age is not None else None
    
    for question in questions:
        age_match = True
        gender_match = True
        
        if target_age_group:
            age_match = (question.age_group == target_age_group or question.age_group == "All")
        
        if gender:
            gender_match = (question.gender == "Both" or 
                           (question.gender and question.gender.lower() == gender.lower()))
        
        if age_match and gender_match:
            filtered_questions.append(question)
    
    return filtered_questions