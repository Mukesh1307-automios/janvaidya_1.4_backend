from fastapi import FastAPI
from api.routes.user import login
from api.routes.user import get_protocols
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



