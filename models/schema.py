from pydantic import BaseModel, EmailStr, StringConstraints, Field, field_validator
from typing import Optional, List, Literal
from typing import Optional, Dict, Any, List, TypedDict,Union
from typing_extensions import Annotated

import re


class DoctorRegistration(BaseModel):
    email: str
    password: Annotated[str, StringConstraints(min_length=6)]
    name: Annotated[str, StringConstraints(min_length=2, max_length=50)]
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        
        if not re.search(r'[@$!%*?&#]', v):
            raise ValueError('Password must contain at least one special character (@$!%*?&#)')
        
        if not re.match(r'^[A-Za-z\d@$!%*?&#]+$', v):
            raise ValueError('Password can only contain letters, digits, and special characters (@$!%*?&#)')
        
        return v


class Response(BaseModel):
    message: str
    user_id: int
    
class LoginValidation(BaseModel):
    email: str
    password: Annotated[str, StringConstraints(min_length=6)]
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        
        if not re.search(r'[@$!%*?&#]', v):
            raise ValueError('Password must contain at least one special character (@$!%*?&#)')
        
        if not re.match(r'^[A-Za-z\d@$!%*?&#]+$', v):
            raise ValueError('Password can only contain letters, digits, and special characters (@$!%*?&#)')
        
        return v
    
class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_at: str
    user_exists: bool
    user_details: dict


class Option(BaseModel):
    option_id: int
    opt_value: str

class Question(BaseModel):
    q_id: int
    question: str
    options: List[Option]
    age_group: str = "All"
    gender: Literal["Male", "Female", "Both"] = "Both"


#----------------------------AI ADDITIONAL QUESTIONS-----------------------------------


# ===================== WORKFLOW STATE =====================

class QuestionGenerationState(TypedDict):
    """State for the additional question generation workflow"""
    protocol_id: int
    doctor_id: int
    num_questions: int
    existing_questions: List[Dict[str, Any]]
    medical_context: str
    llm_prompt: str
    generated_response: str
    parsed_questions: List[Question]
    stored_questions: List[Dict[str, Any]]
    next_q_id: int
    error: Optional[str]
    metadata: Dict[str, Any]

# ===================== MODELS =====================

class GenerateQuestionsRequest(BaseModel):
    protocol_id: int
    doctor_id: int
    num_questions: Optional[int] = 5



#--------------------FULLY AI GENERATION--------------------------


class Options(BaseModel):
    option_id: int = Field(description="Unique identifier for the option")
    opt_value: str = Field(description="Text value of the option")

class QuestionResponse(BaseModel):
    q_id: int = Field(description="Unique question ID")
    q_tag: str = Field(description="Question category tag")
    question: str = Field(description="The diagnostic question")
    options: List[Options] = Field(description="List of answer options")
    age_group: str = Field(description="Age group this question applies to")
    gender: str = Field(description="Gender this question applies to")

class QuestionSetResponse(BaseModel):
    questions: List[QuestionResponse]
    total_questions: int = Field(description="Total number of questions generated")
    medical_condition: str = Field(description="Medical condition these questions are for")
    protocol_id: int = Field(description="Protocol ID")
    doctor_id: int = Field(description="Doctor ID")
    generated_at: str = Field(description="Timestamp of generation")
    session_id: str = Field(description="Session ID for tracking")

class MedicalConditionRequest(BaseModel):
    protocol_id: int = Field(description="Protocol ID to associate questions with")
    doctor_id: int = Field(description="Doctor ID to associate questions with")

class FilterRequest(BaseModel):
    protocol_id: int = Field(description="Protocol ID to filter")
    doctor_id: int = Field(description="Doctor ID to filter")
    age: Optional[Union[int, str]] = None
    gender: Optional[str] = Field(default=None, description="Patient gender (Male/Female)", example="Male")