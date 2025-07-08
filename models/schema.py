from pydantic import BaseModel, EmailStr, StringConstraints, Field, field_validator
from typing import Optional, Dict, Any, List
from typing_extensions import Annotated
from datetime import datetime
from sqlalchemy import Column, Integer, String, Date, DateTime, Boolean, ForeignKey, LargeBinary
from sqlalchemy.orm import relationship 
from sqlalchemy.ext.declarative import declarative_base
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