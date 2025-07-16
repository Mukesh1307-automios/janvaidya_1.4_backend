# routes/question.py
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
from sqlalchemy import text
from db.database import get_db
from utils.helper import get_current_user
from models.schema import QuestionCreate, QuestionUpdate

router = APIRouter()


@router.post("/question/add")
async def add_question(question_data: QuestionCreate, user: dict = Depends(get_current_user)):
    try:
        with get_db() as conn:
        # Get protocol_id from name
            result = conn.execute(text("SELECT id FROM medapp.protocols WHERE name = :name"), {"name": question_data.protocol_name}).fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="Protocol not found")

            protocol_id = result.id
            doc_id = user["user_id"]

            # Insert question
            insert_q = text("""
                INSERT INTO medapp.questions (q_tag, qtn_doc_id, protocol_id, age_group, gender, question)
                VALUES (:q_tag, :qtn_doc_id, :protocol_id, :age_group, :gender, :question)
                RETURNING q_id
            """)
            #q_tag = f"Q_{doc_id}_{protocol_id}"  # Optional logic to generate tag
            result = conn.execute(insert_q, {
                "q_tag": question_data.q_tag,
                "qtn_doc_id": doc_id,
                "protocol_id": protocol_id,
                "age_group": question_data.age_group,
                "gender": question_data.gender,
                "question": question_data.question
            })
            q_id = result.scalar_one()

            # Insert options
            insert_opt = text("""
                INSERT INTO medapp.options (protocol_id, opt_doc_id, opt_q_id, option_value)
                VALUES (:protocol_id, :opt_doc_id, :opt_q_id, :option_value)
            """)
            for opt in question_data.options:
                conn.execute(insert_opt, {
                    "protocol_id": protocol_id,
                    "opt_doc_id": doc_id,
                    "opt_q_id": q_id,
                    "option_value": opt
                })

            return {"message": "Question and options added successfully", "q_id": q_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/question/modification/{q_id}")
async def update_question(q_id: int, data: QuestionUpdate, user: dict = Depends(get_current_user)):
    try:
        with get_db() as conn:
            if data.question or data.age_group or data.gender:
                update_fields = []
                values = {"q_id": q_id}
                if data.question:
                    update_fields.append("question = :question")
                    values["question"] = data.question
                if data.age_group:
                    update_fields.append("age_group = :age_group")
                    values["age_group"] = data.age_group
                if data.gender:
                    update_fields.append("gender = :gender")
                    values["gender"] = data.gender

                update_query = f"UPDATE medapp.questions SET {', '.join(update_fields)} WHERE q_id = :q_id"
                conn.execute(text(update_query), values)

            if data.options:
                # Delete existing options for question
                conn.execute(text("DELETE FROM medapp.options WHERE opt_q_id = :q_id"), {"q_id": q_id})
                # Get protocol and doc_id
                q_row = conn.execute(text("SELECT protocol_id, qtn_doc_id FROM medapp.questions WHERE q_id = :q_id"), {"q_id": q_id}).fetchone()
                if not q_row:
                    raise HTTPException(status_code=404, detail="Question not found")
                for opt in data.options:
                    conn.execute(text("""
                        INSERT INTO medapp.options (protocol_id, opt_doc_id, opt_q_id, option_value)
                        VALUES (:protocol_id, :opt_doc_id, :opt_q_id, :option_value)
                    """), {
                        "protocol_id": q_row.protocol_id,
                        "opt_doc_id": q_row.qtn_doc_id,
                        "opt_q_id": q_id,
                        "option_value": opt
                    })

            return {"message": "Question and options updated successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/question/delete/{q_id}")
async def delete_question(q_id: int, user: dict = Depends(get_current_user)):
    try:
        with get_db() as conn:
        # First delete options to maintain FK constraint
            conn.execute(text("DELETE FROM medapp.options WHERE opt_q_id = :q_id"), {"q_id": q_id})
            conn.execute(text("DELETE FROM medapp.questions WHERE q_id = :q_id"), {"q_id": q_id})
            return {"message": "Question and options deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
