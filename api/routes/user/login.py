
from fastapi import FastAPI, HTTPException, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, StringConstraints
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine, text
from models.schema import DoctorRegistration, Response, LoginValidation, TokenResponse
from core.config import settings
from db.database import get_db
from typing import Annotated, List
import logging
from jose import jwt

logger = logging.getLogger(__name__)


def create_access_token(data: dict) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=settings.ACCESS_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database engine
engine = create_engine(settings.DATABASE_URL)



router = APIRouter(prefix="/user/auth", tags=["Authentication"])


@router.post("/register", response_model=Response)
async def register_user(user: DoctorRegistration):
    try:
        with get_db() as conn:
            # Check if email already exists
            user_details = conn.execute(
                text("SELECT doc_id, name FROM doctors WHERE email = :email"),
                {"email": user.email}
            ).fetchone()

            if user_details:
                return {"message": "User already registered", "user_id": user_details.doc_id}

            # Insert new doctor (name + email) into doctors table
            insert_doctor_query = text("""
                INSERT INTO doctors (name, email)
                VALUES (:name, :email)
                RETURNING doc_id
            """)
            result = conn.execute(insert_doctor_query, {
                "name": user.name,
                "email": user.email
            })
            doctor_id = result.fetchone()[0]

            # Insert password into doctors_password table
            insert_password_query = text("""
                INSERT INTO doctors_password (doc_id, password)
                VALUES (:doc_id, :password)
            """)
            conn.execute(insert_password_query, {
                "doc_id": doctor_id,
                "password": user.password
            })

            return {"message": "User registered successfully", "user_id": doctor_id}

    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during registration."
        )

        
        
@router.post("/login", response_model=TokenResponse)
async def login(validation: LoginValidation):
    """Verify password and return JWT with user status"""
    try:
        with get_db() as db:
            # Join doctors and doctors_password for login check
            user = db.execute(
                text("""
                    SELECT d.doc_id, d.name, d.email
                    FROM doctors d
                    JOIN doctors_password dp ON d.doc_id = dp.doc_id
                    WHERE d.email = :email AND dp.password = :password
                """),
                {"email": validation.email, "password": validation.password}
            ).fetchone()

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )

            # Generate JWT token
            token_data = {
                "sub": str(user.doc_id),
                "email": user.email
            }
            access_token = create_access_token(token_data)
            expire_time = datetime.now(timezone.utc) + timedelta(days=settings.ACCESS_TOKEN_EXPIRE_DAYS)

            user_exists = bool(user.name and user.name.strip())

            user_details = {
                "user_id": user.doc_id,
                "name": user.name,
                "email": user.email,
            } if user_exists else {
                "user_id": user.doc_id
            }

            return TokenResponse(
                access_token=access_token,
                token_type="bearer",
                expires_at=expire_time.isoformat(),
                user_exists=user_exists,
                user_details=user_details
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to authenticate user"
        )
