from fastapi import FastAPI, HTTPException, status, APIRouter
from sqlalchemy import create_engine, text
from models.schema import DoctorRegistration, Response, LoginValidation, TokenResponse
from db.database import get_db
from typing import Annotated, List


router = APIRouter(prefix='/user', tags=['Get Protocols'])
@router.get("/protocols", response_model=List[str])
def list_protocols():
    """
    Returns a unique list of all protocol names.
    """
    try:
        with get_db() as conn:
            result = conn.execute(text("SELECT DISTINCT name FROM protocols ORDER BY name ASC"))
            protocol_names = [row[0] for row in result.fetchall()]
        return protocol_names
    except Exception as e:
        # Log is already handled in get_db
        raise HTTPException(status_code=500, detail="Failed to fetch protocol list")