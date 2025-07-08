from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from core.config import settings
import logging
from sqlalchemy.sql import text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

@contextmanager
def get_db():
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            yield conn
            trans.commit()
            logger.info("Transaction committed.")
        except Exception as e:
            trans.rollback()
            logger.error(f"Transaction rolled back: {e}")
            raise