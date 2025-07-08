from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List
import os
from typing import ClassVar
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "Medical Diagnosis API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_DAYS: int = os.getenv("ACCESS_TOKEN_EXPIRE_DAYS")
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS")    

    class Config:
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
