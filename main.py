from fastapi import FastAPI
from api.routes.user import login
from api.routes.user import get_protocols
from api.routes.user.fileupload import upload
from api.routes.user.AI_qna import additional_questions
from api.routes.user.AI_qna import fully_ai_generation
from api.routes.user import diagnosis_pred
from api.routes.user import questions
from db.database import Base, engine
from fastapi.middleware.cors import CORSMiddleware

 
app = FastAPI(
    title="Doctor Registration API",
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
Base.metadata.create_all(bind=engine)

app.include_router(login.router)
app.include_router(get_protocols.router)
app.include_router(upload.router)
app.include_router(additional_questions.router)
app.include_router(fully_ai_generation.router)
app.include_router(diagnosis_pred.router)
app.include_router(questions.router)



