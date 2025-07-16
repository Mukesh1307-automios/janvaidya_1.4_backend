from fastapi import FastAPI, UploadFile, File, Form, HTTPException,status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Literal
import tempfile
import os
from pathlib import Path
import logging
from fastapi.responses import JSONResponse
from sqlalchemy import text
from models.schema import Question , Option
from db.database import get_db
from utils.helper import get_file_type,logger
# Import your processors
from api.routes.user.fileupload.text_question_processor import TextQuestionProcessor, has_questions_and_options
from api.routes.user.fileupload.medical_csv_processor import MedicalCSVQuestionProcessor
from api.routes.user.fileupload.medical_document_processor import MedicalDocumentQuestionGenerator


# ===================== API ENDPOINTS =====================

router = APIRouter(prefix='', tags=['File Upload'])
@router.post("/FileUpload", response_model=List[Question])
async def process_content(
    prompt: str = Form(..., description="Your request/prompt - can contain file processing instructions OR questions with options directly"),
    file: Optional[UploadFile] = File(None, description="Optional file upload (PDF, TXT, DOC, DOCX, CSV, XLSX, XLS)")
):
    """
    Process content and return structured questions in the specified format.
    """
    
    try:
        logger.info(f"📥 Received request - File: {file.filename if file else 'None'}, Prompt length: {len(prompt)}")
        
        # Validate input
        if not prompt or not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        # Initialize text processor
        text_processor = TextQuestionProcessor()
        
        # Determine processing type
        will_process_file = file is not None
        will_process_text = not will_process_file and has_questions_and_options(prompt)
        
        logger.info(f"🔍 Processing type: file={will_process_file}, text={will_process_text}")
        
        # Process content
        if will_process_file:
            logger.info(f"📁 Processing file: {file.filename}")
            result = await process_file_upload(file, prompt)
            logger.info(f"📁 File processing status: {result['status']}")
            
            if result["status"] != "success":
                raise HTTPException(status_code=500, detail=result.get("error", "Processing failed"))
            
            # Log response info
            raw_response = result["response"]
            logger.info(f"📄 Raw response length: {len(raw_response)} characters")
            logger.info(f"📄 Raw response preview: {raw_response[:100]}...")
            
            # Count questions in raw response
            raw_question_count = raw_response.count("Question ")
            logger.info(f"📊 Questions in raw response: {raw_question_count}")
            
            # Parse questions
            logger.info(f"🔄 Parsing questions with text processor...")
            questions = text_processor.extract_questions_from_response(raw_response)
            logger.info(f"📊 Text processor extracted: {len(questions)} questions")
            
            # If we lost questions, try simpler parsing
            if len(questions) < raw_question_count:
                logger.warning(f"⚠️  Text processor only got {len(questions)}/{raw_question_count} questions")
                logger.info(f"🔄 Trying alternative parsing...")
                
                # Alternative: Simple regex-based parsing
                import re
                question_blocks = re.split(r'\n(?=Question \d+:)', raw_response)
                alternative_questions = []
                
                for i, block in enumerate(question_blocks):
                    if 'Question' in block:
                        lines = [line.strip() for line in block.split('\n') if line.strip()]
                        if lines:
                            # Extract question
                            question_line = lines[0]
                            question_text = re.sub(r'^Question \d+:\s*', '', question_line).strip()
                            
                            # Extract options
                            options = []
                            option_id = 1
                            for line in lines[1:]:
                                if re.match(r'^[A-Z]\)', line):
                                    option_text = re.sub(r'^[A-Z]\)\s*', '', line).strip()
                                    options.append({
                                        "option_id": option_id,
                                        "opt_value": option_text
                                    })
                                    option_id += 1
                            
                            if question_text and len(options) >= 2:
                                alternative_questions.append({
                                    "q_id": len(alternative_questions) + 1,
                                    "question": question_text,
                                    "options": options,
                                    "age_group": "All",
                                    "gender": "Both"
                                })
                
                logger.info(f"🔄 Alternative parsing got: {len(alternative_questions)} questions")
                
                if len(alternative_questions) > len(questions):
                    logger.info(f"✅ Using alternative parsing results")
                    # Convert to Question objects
                    questions = []
                    for q_data in alternative_questions:
                        options = [Option(**opt) for opt in q_data["options"]]
                        questions.append(Question(
                            q_id=q_data["q_id"],
                            question=q_data["question"],
                            options=options,
                            age_group=q_data["age_group"],
                            gender=q_data["gender"]
                        ))
            
        else:
            logger.info(f"📝 Processing text directly")
            questions = text_processor.parse_direct_input(prompt)
            logger.info(f"📊 Direct text processing got: {len(questions)} questions")
        
        if not questions:
            raise HTTPException(status_code=404, detail="No questions found in the content")
        
        # ===================== STORE IN DATABASE =====================
        logger.info(f"💾 Storing {len(questions)} questions in database...")

        try:
            with get_db() as conn:
                stored_questions = []
                
                for q in questions:
                    # Use the original q_id from the Question object
                    original_q_id = q.q_id
                    
                    # Insert question into questions table (based on your schema)
                    insert_question_query = text("""
                        INSERT INTO medapp.questions (q_id, q_tag, qtn_doc_id, protocol_id, age_group, gender, question)
                        VALUES (:q_id, :q_tag, :qtn_doc_id, :protocol_id, :age_group, :gender, :question)
                    """)
                    
                    # Get q_tag from question object or set default
                    q_tag = getattr(q, 'q_tag', 'general')
                    
                    conn.execute(insert_question_query, {
                        "q_id": original_q_id,  # Use original q_id from Question object
                        "q_tag": q_tag,
                        "qtn_doc_id": 7,  # Hardcoded as requested (equivalent to doctor_id)
                        "protocol_id": 1,  # Hardcoded as requested
                        "age_group": q.age_group,
                        "gender": q.gender,
                        "question": q.question
                    })
                    
                    logger.info(f"📝 Inserted question with original q_id: {original_q_id}")
                    
                    # Insert options into options table (based on your schema)
                    for option in q.options:
                        insert_option_query = text("""
                            INSERT INTO medapp.options (protocol_id, opt_doc_id, opt_q_id, option_value)
                            VALUES (:protocol_id, :opt_doc_id, :opt_q_id, :option_value)
                        """)
                        
                        conn.execute(insert_option_query, {
                            "protocol_id": 1,  # Same protocol_id as question
                            "opt_doc_id": 7,  # Same as qtn_doc_id (equivalent to doctor_id)
                            "opt_q_id": original_q_id,  # Reference to the question
                            "option_value": option.opt_value
                        })
                    
                    logger.info(f"📝 Inserted {len(q.options)} options for question {original_q_id}")
                    
                    # Store question data for response
                    stored_questions.append({
                        "q_id": original_q_id,
                        "question": q.question,
                        "options": [{"option_id": opt.option_id, "opt_value": opt.opt_value} for opt in q.options],
                        "q_tag": q_tag,
                        "age_group": q.age_group,
                        "gender": q.gender
                    })
                
                logger.info(f"✅ Successfully stored all questions and options in database")
                
                # Return the stored questions with original q_ids
                result_data = stored_questions
                
        except Exception as db_error:
            logger.error(f"🚨 Database storage error: {db_error}")
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Database storage failed: {str(db_error)}")
        
        logger.info(f"✅ Returning {len(result_data)} questions")
        
        # Check response size
        import json
        response_json = json.dumps(result_data)
        response_size = len(response_json)
        logger.info(f"📊 Response size: {response_size} bytes ({response_size/1024:.1f} KB)")
        
        # If response is very large, limit it
        if response_size > 2 * 1024 * 1024:  # 2MB limit
            logger.warning(f"⚠️  Response too large ({response_size/1024/1024:.1f} MB), truncating to first 20 questions")
            result_data = result_data[:20]
        
        return JSONResponse(content=result_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🚨 Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ===================== FILE PROCESSING =====================

async def process_file_upload(uploaded_file: UploadFile, user_prompt: str):
    """Process file upload"""
    file_type = get_file_type(uploaded_file.filename)
    
    if file_type == 'unsupported':
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.filename).suffix) as tmp_file:
            content = await uploaded_file.read()
            tmp_file.write(content)
            temp_file_path = tmp_file.name
        
        logger.info(f"Processing {file_type} file: {uploaded_file.filename}")
        
        # Process based on file type
        if file_type == 'data':
            processor = MedicalCSVQuestionProcessor()
            result = processor.process(temp_file_path, user_prompt)
        else:
            processor = MedicalDocumentQuestionGenerator()
            result = processor.process(temp_file_path, user_prompt)
        
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        return result
        
    except Exception as e:
        # Clean up on error
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise e