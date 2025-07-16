from fastapi import HTTPException, APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import logging
from datetime import datetime
import uuid
from utils.helper import fix_common_json_errors, determine_age_group, generate_questions_with_llm, filter_questions_by_demographics

# Database imports
from sqlalchemy import text
from models.schema import FilterRequest, Options, MedicalConditionRequest, QuestionSetResponse, QuestionResponse
from db.database import get_db
from utils.helper import logger

# Import prompts
from api.routes.user.AI_qna.prompt import QUESTIONS_ONLY_PROMPT

# ================================
# ROUTER SETUP
# ================================

router = APIRouter(prefix='', tags=['Questions Management'])

# ================================
# QUESTION APIs
# ================================

@router.post("/GenerateAI_Questions", response_model=QuestionSetResponse)
async def generate_medical_questions(request: MedicalConditionRequest):
    """Generate medical questions for a given condition and store in database."""
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

        # Generate questions using LLM
        session_id = str(uuid.uuid4())
        questions_data = await generate_questions_with_llm(request.protocol_id, request.doctor_id)
        
        # Get next available q_id
        with get_db() as conn:
            max_q_id_result = conn.execute(
                text("SELECT COALESCE(MAX(q_id), 0) FROM medapp.questions")
            ).scalar()
            next_q_id = max_q_id_result + 1

        questions = []
        stored_questions = []
        
        # Process and store each question
        with get_db() as conn:
            for idx, q_data in enumerate(questions_data["questions"]):
                current_q_id = next_q_id + idx
                
                # Create Question object
                options = []
                for opt_data in q_data["options"]:
                    option = Options(
                        option_id=opt_data["option_id"],
                        opt_value=opt_data["opt_value"]
                    )
                    options.append(option)

                question = QuestionResponse(
                    q_id=current_q_id,
                    q_tag="ai_generated",  # Set as ai_generated
                    question=q_data["question"],
                    options=options,
                    age_group=q_data["age_group"],
                    gender=q_data["gender"]
                )
                questions.append(question)

                # Store question in database
                insert_question_query = text("""
                    INSERT INTO medapp.questions (q_id, q_tag, qtn_doc_id, protocol_id, age_group, gender, question)
                    VALUES (:q_id, :q_tag, :qtn_doc_id, :protocol_id, :age_group, :gender, :question)
                """)
                
                conn.execute(insert_question_query, {
                    "q_id": current_q_id,
                    "q_tag": "ai_generated",
                    "qtn_doc_id": request.doctor_id,
                    "protocol_id": request.protocol_id,
                    "age_group": question.age_group,
                    "gender": question.gender,
                    "question": question.question
                })
                
                # Store options in database
                for option in question.options:
                    insert_option_query = text("""
                        INSERT INTO medapp.options (protocol_id, opt_doc_id, opt_q_id, option_value)
                        VALUES (:protocol_id, :opt_doc_id, :opt_q_id, :option_value)
                    """)
                    
                    conn.execute(insert_option_query, {
                        "protocol_id": request.protocol_id,
                        "opt_doc_id": request.doctor_id,
                        "opt_q_id": current_q_id,
                        "option_value": option.opt_value
                    })
                
                # Store for response
                stored_questions.append({
                    "q_id": current_q_id,
                    "q_tag": "ai_generated",
                    "question": question.question,
                    "options": [{"option_id": opt.option_id, "opt_value": opt.opt_value} for opt in question.options],
                    
                    "age_group": question.age_group,
                    "gender": question.gender
                })
                
                logger.info(f"Stored question {current_q_id}: {question.question}")

        logger.info(f"Successfully generated and stored {len(questions)} questions")

        return QuestionSetResponse(
            questions=questions,
            total_questions=len(questions),
            protocol_id=request.protocol_id,
            doctor_id=request.doctor_id,
            generated_at=datetime.now().isoformat(),
            session_id=session_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in generate_medical_questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/GetQuestions")
async def get_questions(
    protocol_id: int = Query(..., description="Protocol ID"),
    doctor_id: int = Query(..., description="Doctor ID")
):
    """Get all questions for a specific protocol and doctor."""
    try:
        with get_db() as conn:
            # Get all questions (both general and ai_generated)
            questions_query = text("""
                SELECT q.q_id, q.question, q.age_group, q.gender, q.q_tag
                FROM medapp.questions q 
                WHERE q.protocol_id = :protocol_id AND q.qtn_doc_id = :doctor_id
                ORDER BY q.q_id
            """)
            
            questions_result = conn.execute(questions_query, {
                "protocol_id": protocol_id,
                "doctor_id": doctor_id
            }).fetchall()
            
            if not questions_result:
                return JSONResponse(content={"message": "No questions found", "questions": []})
            
            # Get options for each question
            questions_with_options = []
            for q_row in questions_result:
                options_query = text("""
                    SELECT option_value FROM medapp.options 
                    WHERE protocol_id = :protocol_id AND opt_doc_id = :doctor_id AND opt_q_id = :q_id
                    ORDER BY id
                """)
                
                options_result = conn.execute(options_query, {
                    "protocol_id": protocol_id,
                    "doctor_id": doctor_id,
                    "q_id": q_row.q_id
                }).fetchall()
                
                questions_with_options.append({
                    "q_id": q_row.q_id,
                    "question": q_row.question,
                    "age_group": q_row.age_group,
                    "gender": q_row.gender,
                    "q_tag": q_row.q_tag,
                    "options": [opt.option_value for opt in options_result]
                })
            
            return JSONResponse(content={
                "protocol_id": protocol_id,
                "doctor_id": doctor_id,
                "total_questions": len(questions_with_options),
                "questions": questions_with_options
            })
            
    except Exception as e:
        logger.error(f"Error retrieving questions: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving questions: {str(e)}")

@router.get("/GetQuestionById/{q_id}", response_model=QuestionResponse)
async def get_question_by_id(
    q_id: int,
    protocol_id: int = Query(..., description="Protocol ID"),
    doctor_id: int = Query(..., description="Doctor ID")
):
    """Get a specific question by q_id."""
    try:
        with get_db() as conn:
            # Get specific question
            question_query = text("""
                SELECT q.q_id, q.question, q.age_group, q.gender, q.q_tag
                FROM medapp.questions q 
                WHERE q.q_id = :q_id AND q.protocol_id = :protocol_id AND q.qtn_doc_id = :doctor_id
            """)
            
            question_result = conn.execute(question_query, {
                "q_id": q_id,
                "protocol_id": protocol_id,
                "doctor_id": doctor_id
            }).fetchone()
            
            if not question_result:
                raise HTTPException(status_code=404, detail=f"Question with q_id {q_id} not found")
            
            # Get options for the question
            options_query = text("""
                SELECT option_value FROM medapp.options 
                WHERE protocol_id = :protocol_id AND opt_doc_id = :doctor_id AND opt_q_id = :q_id
                ORDER BY id
            """)
            
            options_result = conn.execute(options_query, {
                "protocol_id": protocol_id,
                "doctor_id": doctor_id,
                "q_id": q_id
            }).fetchall()
            
            # Build options list
            options = []
            for idx, opt in enumerate(options_result, 1):
                options.append(Options(
                    option_id=idx,
                    opt_value=opt.option_value
                ))
            
            return QuestionResponse(
                q_id=question_result.q_id,
                q_tag=question_result.q_tag,
                question=question_result.question,
                options=options,
                age_group=question_result.age_group,
                gender=question_result.gender
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving question {q_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving question: {str(e)}")

@router.post("/FilterQuestions")
async def get_filtered_questions(request: FilterRequest):
    """Get questions filtered by age and gender demographics."""
    try:
        with get_db() as conn:
            # Build dynamic query based on filters
            base_query = """
                SELECT q.q_id, q.question, q.age_group, q.gender, q.q_tag
                FROM medapp.questions q 
                WHERE q.protocol_id = :protocol_id AND q.qtn_doc_id = :doctor_id
            """
            
            params = {
                "protocol_id": request.protocol_id,
                "doctor_id": request.doctor_id
            }
            
            # Add age group filter if provided
            # Add age group filter if provided
            if request.age is not None:
                if isinstance(request.age, str) and request.age.lower() == "all":
                    # If age is "All", don't add any age filter (return all age groups)
                    pass
                else:
                    # Handle integer age
                    if isinstance(request.age, str):
                        try:
                            age_int = int(request.age)
                        except ValueError:
                            raise HTTPException(status_code=400, detail="Invalid age format")
                    else:
                        age_int = request.age
                        
                    target_age_group = determine_age_group(age_int)
                    base_query += " AND (q.age_group = :age_group OR q.age_group = 'All')"
                    params["age_group"] = target_age_group
            
            # Add gender filter if provided
            if request.gender is not None:
                base_query += " AND (q.gender = :gender OR q.gender = 'Both')"
                params["gender"] = request.gender
            
            base_query += " ORDER BY q.q_id"
            
            questions_result = conn.execute(text(base_query), params).fetchall()
            
            if not questions_result:
                return JSONResponse(content={"message": "No filtered questions found", "questions": []})
            
            # Get options for each question
            questions_with_options = []
            for q_row in questions_result:
                options_query = text("""
                    SELECT option_value FROM medapp.options 
                    WHERE protocol_id = :protocol_id AND opt_doc_id = :doctor_id AND opt_q_id = :q_id
                    ORDER BY id
                """)
                
                options_result = conn.execute(options_query, {
                    "protocol_id": request.protocol_id,
                    "doctor_id": request.doctor_id,
                    "q_id": q_row.q_id
                }).fetchall()
                
                questions_with_options.append({
                    "q_id": q_row.q_id,
                    "question": q_row.question,
                    "age_group": q_row.age_group,
                    "gender": q_row.gender,
                    "q_tag": q_row.q_tag,
                    "options": [opt.option_value for opt in options_result]
                })
            
            logger.info(f"Filtered {len(questions_with_options)} questions for protocol {request.protocol_id}, doctor {request.doctor_id}")
            
            return JSONResponse(content={
                "protocol_id": request.protocol_id,
                "doctor_id": request.doctor_id,
                "filters_applied": {
                    "age": request.age,
                    "gender": request.gender,
                    "age_group": determine_age_group(request.age) if request.age else None
                },
                "total_questions": len(questions_with_options),
                "questions": questions_with_options
            })
    
    except Exception as e:
        logger.error(f"Error filtering questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/GetAllQuestions")
async def get_all_questions():
    """Get all questions across all protocols and doctors."""
    try:
        with get_db() as conn:
            # Get all questions
            questions_query = text("""
                SELECT q.q_id, q.question, q.age_group, q.gender, q.q_tag, q.protocol_id, q.qtn_doc_id
                FROM medapp.questions q 
                ORDER BY q.protocol_id, q.qtn_doc_id, q.q_id
            """)
            
            questions_result = conn.execute(questions_query).fetchall()
            
            if not questions_result:
                return JSONResponse(content={"message": "No questions found", "questions": []})
            
            # Get options for each question
            questions_with_options = []
            for q_row in questions_result:
                options_query = text("""
                    SELECT option_value FROM medapp.options 
                    WHERE protocol_id = :protocol_id AND opt_doc_id = :doctor_id AND opt_q_id = :q_id
                    ORDER BY id
                """)
                
                options_result = conn.execute(options_query, {
                    "protocol_id": q_row.protocol_id,
                    "doctor_id": q_row.qtn_doc_id,
                    "q_id": q_row.q_id
                }).fetchall()
                
                questions_with_options.append({
                    "q_id": q_row.q_id,
                    "question": q_row.question,
                    "age_group": q_row.age_group,
                    "gender": q_row.gender,
                    "q_tag": q_row.q_tag,
                    "protocol_id": q_row.protocol_id,
                    "doctor_id": q_row.qtn_doc_id,
                    "options": [opt.option_value for opt in options_result]
                })
            
            return JSONResponse(content={
                "total_questions": len(questions_with_options),
                "questions": questions_with_options
            })
            
    except Exception as e:
        logger.error(f"Error retrieving all questions: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving questions: {str(e)}")