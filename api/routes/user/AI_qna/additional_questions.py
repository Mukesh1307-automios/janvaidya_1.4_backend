import os
import logging
from typing import List, Dict, Any, Optional, TypedDict
from pathlib import Path
import json
import re

# FastAPI imports
from fastapi import FastAPI, HTTPException, APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# LangChain imports
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

# LangGraph imports
from langgraph.graph import StateGraph, END



# Database imports
from sqlalchemy import text
from models.schema import GenerateQuestionsRequest , QuestionGenerationState , Question, Option
from db.database import get_db
from utils.helper import logger


# Global LLM cache to avoid reloading
_global_llm_cache = None



# ===================== PROMPT TEMPLATE =====================

class MedicalQuestionPromptTemplate:
    """Enhanced medical question generation prompting template"""
    
    @staticmethod
    def get_template() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert medical professional creating clinical interview questions for patient assessment.

CRITICAL RULES:
- Generate questions that are SPECIFIC to the given medical protocol
- Generate questions that are COMPLETELY DIFFERENT from existing questions (do not duplicate)
- Use clear, patient-friendly language
- Create clinically relevant and meaningful options
- Each question must have 2-3 options only
- Use numbered format (1, 2, 3) - NEVER use A, B, C, D or more than 3 options
- Questions should be progressive and comprehensive for the specific protocol
- Ensure options are mutually exclusive and comprehensive
- Stay focused on the protocol's medical domain
- Follow the EXACT format specified below"""),
            
            ("human", """You are creating additional questions for the "{protocol_name}" protocol. Generate {num_questions} new questions that are COMPLETELY DIFFERENT from the existing ones.

EXISTING QUESTIONS IN THIS PROTOCOL:
{existing_questions_context}

CRITICAL INSTRUCTIONS:
1. Look at ALL the existing questions above
2. DO NOT create similar or duplicate questions
3. Focus on the "{protocol_name}" protocol
4. Generate {num_questions} NEW questions that cover DIFFERENT aspects
5. Avoid these already covered types: {covered_types}

REQUIRED FORMAT - Follow this EXACT format for each question:

Question 1: [Your NEW question for {protocol_name}]?
   1. [Specific option]
   2. [Specific option]
   3. [Specific option]
   

Question 2: [Your NEW question for {protocol_name}]?
   1. [Specific option]
   2. [Specific option]
   3. [Specific option]
   

CRITICAL FORMAT REQUIREMENTS:
- Start each question with "Question X:" (where X is the number)
- End each question with a question mark
- Use only 2-3 numbered options (1, 2, 3 maximum)
- No extra text or explanations
- No markdown formatting like ** or __
- Each option on a new line starting with the number

Generate exactly {num_questions} questions following this format.

CRITICAL: 
- All questions must be relevant to "{protocol_name}"
- Do NOT duplicate ANY existing questions
- Create questions about DIFFERENT aspects than what already exists
- Focus on new areas not covered by existing questions""")
        ])

# ===================== LLM SERVICE =====================

class MedicalLLMService:
    """Enhanced LLM service with caching and proper configuration"""
    
    def __init__(self):
        global _global_llm_cache
        
        if _global_llm_cache is None:
            logger.info("Initializing Ollama LLM for medical question generation...")
            _global_llm_cache = OllamaLLM(
                model="llama3.1:8b",
                temperature=0.2,  # Lower temperature for more consistent formatting
                num_predict=4000,  # More tokens for multiple questions
                top_p=0.8,
                repeat_penalty=1.1,
                stop=["\n\n\n"]  # Stop at triple newlines to prevent rambling
            )
            logger.info("LLM initialized successfully")
        else:
            logger.info("Using cached LLM instance")
            
        self.llm = _global_llm_cache
        self.prompt_template = MedicalQuestionPromptTemplate.get_template()
    
    def generate_questions(self, existing_questions: List[Dict], medical_context: Dict, num_questions: int) -> str:
        """Generate additional questions using protocol name only"""
        try:
            # Format existing questions for context
            existing_context = self._format_existing_questions(existing_questions)
            
            # Format prompt with protocol name only
            formatted_prompt = self.prompt_template.format_messages(
                num_questions=num_questions,
                protocol_name=medical_context.get("protocol_name", "Unknown Protocol"),
                existing_questions_context=existing_context,
                covered_types=", ".join(medical_context.get("covered_question_types", []))
            )
            
            logger.info(f"Generating {num_questions} questions for protocol: {medical_context.get('protocol_name')}")
            logger.info(f"Avoiding covered types: {medical_context.get('covered_question_types')}")
            logger.info(f"Total existing questions to avoid: {medical_context.get('total_existing', 0)}")
            
            # Log the actual prompt being sent (truncated for readability)
            prompt_str = str(formatted_prompt)
            logger.info(f"Prompt preview: {prompt_str[:500]}...")
            
            # Generate response
            response = self.llm.invoke(formatted_prompt)
            
            # Extract content
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            # Log full response for debugging
            logger.info(f"LLM Response (full): {result}")
            
            return result
                
        except Exception as e:
            logger.error(f"Error generating questions with LLM: {e}")
            raise
    
    def _format_existing_questions(self, questions: List[Dict]) -> str:
        """Format existing questions for LLM context"""
        context = ""
        for i, q in enumerate(questions, 1):
            context += f"EXISTING QUESTION {i}: {q['question']}\n"
            for j, option in enumerate(q['options'], 1):
                context += f"   {j}. {option['opt_value']}\n"
            context += f"   [Tags: {q.get('q_tag', 'general')}, Age: {q.get('age_group', 'All')}, Gender: {q.get('gender', 'Both')}]\n\n"
        
        return context.strip()

# ===================== QUESTION PARSER =====================

class EnhancedQuestionParser:
    """Enhanced parser with better error handling and validation - based on reference code"""
    
    @staticmethod
    def parse_llm_response(response_text: str, start_q_id: int) -> List[Question]:
        """Parse LLM response into Question objects with multiple format support"""
        questions = []
        
        try:
            # Clean response text
            response_text = response_text.strip()
            logger.info(f"Parsing LLM response: {response_text[:300]}...")
            
            # Try multiple parsing approaches (like reference code)
            
            # Method 1: Split by various question patterns
            question_patterns = [
                r'\n(?=\*\*Question\s+\d+:\*\*)',  # **Question 1:**
                r'\n(?=Question\s+\d+:)',          # Question 1:
                r'\n(?=\d+\.\s)',                  # 1. 
                r'\n(?=Q\d+:)',                    # Q1:
            ]
            
            question_blocks = []
            for pattern in question_patterns:
                blocks = re.split(pattern, response_text)
                if len(blocks) > 1:
                    question_blocks = blocks
                    logger.info(f"Found {len(blocks)} blocks using pattern: {pattern}")
                    break
            
            if not question_blocks:
                # Fallback: split by double newlines
                question_blocks = response_text.split('\n\n')
                logger.info(f"Using fallback splitting, found {len(question_blocks)} blocks")
            
            for block_idx, block in enumerate(question_blocks):
                if not block.strip():
                    continue
                
                # Parse individual question block
                parsed_question = EnhancedQuestionParser._parse_question_block_advanced(
                    block, start_q_id + len(questions)
                )
                
                if parsed_question:
                    questions.append(parsed_question)
                    logger.info(f"Successfully parsed question {len(questions)}: {parsed_question.question[:50]}...")
                else:
                    logger.warning(f"Failed to parse question block {block_idx + 1}: {block[:100]}...")
            
            # If still no questions, try the reference code approach
            if not questions:
                logger.info("Trying reference code parsing approach...")
                questions = EnhancedQuestionParser._parse_with_reference_method(response_text, start_q_id)
            
            logger.info(f"Successfully parsed {len(questions)} valid questions")
            return questions
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return []
    
    @staticmethod
    def _parse_question_block_advanced(block: str, q_id: int) -> Optional[Question]:
        """Advanced parsing for individual question block"""
        try:
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            if not lines:
                return None
            
            # Extract question text - multiple patterns
            question_text = None
            question_patterns = [
                r'\*\*Question\s+\d+:\*\*\s*(.+)',      # **Question 1:** format
                r'Question\s+\d+:\s*(.+)',               # Question 1: format
                r'\d+\.\s*(.+)',                         # 1. format
                r'Q\d+:\s*(.+)',                         # Q1: format
                r'^(.+\?)$',                             # Any line ending with ?
            ]
            
            for line in lines:
                for pattern in question_patterns:
                    match = re.match(pattern, line, re.IGNORECASE)
                    if match:
                        question_text = match.group(1).strip()
                        break
                if question_text:
                    break
            
            if not question_text:
                # Fallback: first line that looks like a question
                for line in lines:
                    if '?' in line and len(line) > 10:
                        question_text = line
                        break
            
            if not question_text:
                logger.warning(f"Could not extract question text from block: {block[:100]}...")
                return None
            
            # Extract options - multiple patterns
            options = []
            option_patterns = [
                r'^\s*(\d+)[\.\)]\s*(.+)',               # 1. or 1)
                r'^\s*([A-D])[\.\)]\s*(.+)',             # A. or A)
                r'^\s*[-*]\s*(.+)',                      # - or *
                r'^\s*([ivxlc]+)[\.\)]\s*(.+)',          # i. or i) (roman numerals)
            ]
            
            option_id = 1
            for line in lines:
                if line == lines[0]:  # Skip question line
                    continue
                    
                option_text = None
                for pattern in option_patterns:
                    match = re.match(pattern, line, re.IGNORECASE)
                    if match:
                        if match.lastindex >= 2:
                            option_text = match.group(2).strip()
                        else:
                            option_text = match.group(1).strip()
                        break
                
                # If no pattern matches but line looks like an option
                if not option_text and len(line) > 3 and not '?' in line:
                    # Clean up potential option text
                    cleaned = re.sub(r'^\W+', '', line).strip()
                    if len(cleaned) > 3:
                        option_text = cleaned
                
                if option_text and len(option_text) > 2:  # Valid option
                    options.append(Option(
                        option_id=option_id,
                        opt_value=option_text
                    ))
                    option_id += 1
                    
                    # Stop if we have 3 options
                    if len(options) >= 3:
                        break
            
            # Validate question
            if not question_text or len(options) < 2 or len(options) > 3:
                logger.warning(f"Invalid question: text='{question_text}', options={len(options)}")
                return None
            
            # Ensure question ends with ?
            if not question_text.endswith('?'):
                question_text += '?'
            
            # Clean question text
            question_text = re.sub(r'\*\*', '', question_text)  # Remove markdown
            question_text = question_text.strip()
            
            return Question(
                q_id=q_id,
                question=question_text,
                options=options,
                age_group="All",
                gender="Both"
            )
            
        except Exception as e:
            logger.error(f"Error parsing question block: {e}")
            return None
    
    @staticmethod
    def _parse_with_reference_method(response_text: str, start_q_id: int) -> List[Question]:
        """Use the reference code parsing method as fallback"""
        try:
            logger.info("Applying reference code parsing method...")
            
            # Alternative: Simple regex-based parsing (from reference code)
            question_blocks = re.split(r'\n(?=Question \d+:)', response_text)
            alternative_questions = []
            
            for i, block in enumerate(question_blocks):
                if 'Question' in block or any(char.isdigit() for char in block[:20]):
                    lines = [line.strip() for line in block.split('\n') if line.strip()]
                    if lines:
                        # Extract question
                        question_line = lines[0]
                        
                        # Multiple question extraction patterns
                        question_patterns = [
                            r'^Question \d+:\s*(.+)',
                            r'^\d+\.\s*(.+)',
                            r'^(.+\?).*',
                        ]
                        
                        question_text = None
                        for pattern in question_patterns:
                            match = re.match(pattern, question_line)
                            if match:
                                question_text = match.group(1).strip()
                                break
                        
                        if not question_text:
                            question_text = question_line.strip()
                        
                        # Extract options
                        options = []
                        option_id = 1
                        for line in lines[1:]:
                            # Support both numbered (1, 2, 3, 4) and lettered (A, B, C, D) options
                            if re.match(r'^\s*[1-4][\.\)]\s*', line) or re.match(r'^\s*[A-D][\.\)]\s*', line):
                                option_text = re.sub(r'^\s*[1-4A-D][\.\)]\s*', '', line).strip()
                                if option_text:
                                    options.append({
                                        "option_id": option_id,
                                        "opt_value": option_text
                                    })
                                    option_id += 1
                        
                        if question_text and len(options) >= 2:
                            alternative_questions.append({
                                "q_id": start_q_id + len(alternative_questions),
                                "question": question_text,
                                "options": options,
                                "age_group": "All",
                                "gender": "Both"
                            })
            
            logger.info(f"Reference method parsed: {len(alternative_questions)} questions")
            
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
            
            return questions
            
        except Exception as e:
            logger.error(f"Reference method parsing failed: {e}")
            return []

# ===================== LANGGRAPH WORKFLOW =====================

class AdditionalQuestionWorkflow:
    """LangGraph-based workflow for additional question generation"""
    
    def __init__(self):
        self.llm_service = MedicalLLMService()
        self.parser = EnhancedQuestionParser()
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(QuestionGenerationState)
        
        # Add nodes
        workflow.add_node("validate_inputs", self._validate_inputs)
        workflow.add_node("retrieve_existing_questions", self._retrieve_existing_questions)
        workflow.add_node("prepare_context", self._prepare_context)
        workflow.add_node("generate_with_llm", self._generate_with_llm)
        workflow.add_node("parse_response", self._parse_response)
        workflow.add_node("store_questions", self._store_questions)
        
        # Define workflow edges
        workflow.set_entry_point("validate_inputs")
        workflow.add_edge("validate_inputs", "retrieve_existing_questions")
        workflow.add_edge("retrieve_existing_questions", "prepare_context")
        workflow.add_edge("prepare_context", "generate_with_llm")
        workflow.add_edge("generate_with_llm", "parse_response")
        workflow.add_edge("parse_response", "store_questions")
        workflow.add_edge("store_questions", END)
        
        return workflow.compile()
    
    def _validate_inputs(self, state: QuestionGenerationState) -> QuestionGenerationState:
        """Validate input parameters"""
        try:
            with get_db() as conn:
                # Verify protocol exists
                protocol_exists = conn.execute(
                    text("SELECT COUNT(*) FROM medapp.protocols WHERE id = :id"), 
                    {"id": state["protocol_id"]}
                ).scalar()
                
                # Verify doctor exists (using doc_id column)
                doctor_exists = conn.execute(
                    text("SELECT COUNT(*) FROM medapp.doctors WHERE doc_id = :doc_id"), 
                    {"doc_id": state["doctor_id"]}
                ).scalar()
                
                if protocol_exists == 0:
                    state["error"] = f"Protocol ID {state['protocol_id']} not found"
                elif doctor_exists == 0:
                    state["error"] = f"Doctor ID {state['doctor_id']} not found"
                else:
                    state["metadata"] = {
                        "protocol_validated": True,
                        "doctor_validated": True,
                        "requested_questions": state["num_questions"]
                    }
                    logger.info(f"Validated protocol_id={state['protocol_id']}, doctor_id={state['doctor_id']}")
                    
        except Exception as e:
            state["error"] = f"Validation error: {e}"
            
        return state
    
    def _retrieve_existing_questions(self, state: QuestionGenerationState) -> QuestionGenerationState:
        """Retrieve existing questions from database"""
        if state.get("error"):
            return state
            
        try:
            with get_db() as conn:
                # Get existing questions
                existing_questions_query = text("""
                    SELECT q.q_id, q.question, q.age_group, q.gender, q.q_tag
                    FROM medapp.questions q 
                    WHERE q.protocol_id = :protocol_id AND q.qtn_doc_id = :doctor_id
                    ORDER BY q.q_id
                """)
                
                questions_result = conn.execute(existing_questions_query, {
                    "protocol_id": state["protocol_id"],
                    "doctor_id": state["doctor_id"]
                }).fetchall()
                
                if not questions_result:
                    state["error"] = f"No existing questions found for protocol_id={state['protocol_id']} and doctor_id={state['doctor_id']}"
                    return state
                
                # Get options for each question
                existing_questions = []
                for q_row in questions_result:
                    options_query = text("""
                        SELECT option_value FROM medapp.options 
                        WHERE protocol_id = :protocol_id AND opt_doc_id = :doctor_id AND opt_q_id = :q_id
                        ORDER BY id
                    """)
                    
                    options_result = conn.execute(options_query, {
                        "protocol_id": state["protocol_id"],
                        "doctor_id": state["doctor_id"],
                        "q_id": q_row.q_id
                    }).fetchall()
                    
                    existing_questions.append({
                        "q_id": q_row.q_id,
                        "question": q_row.question,
                        "age_group": q_row.age_group,
                        "gender": q_row.gender,
                        "q_tag": q_row.q_tag,
                        "options": [{"opt_value": opt.option_value} for opt in options_result]
                    })
                
                state["existing_questions"] = existing_questions
                state["metadata"]["existing_questions_count"] = len(existing_questions)
                
                # Get next available q_id
                max_q_id_result = conn.execute(
                    text("SELECT COALESCE(MAX(q_id), 0) FROM medapp.questions")
                ).scalar()
                state["next_q_id"] = max_q_id_result + 1
                
                logger.info(f"Retrieved {len(existing_questions)} existing questions")
                
        except Exception as e:
            state["error"] = f"Error retrieving existing questions: {e}"
            
        return state
    
    def _prepare_context(self, state: QuestionGenerationState) -> QuestionGenerationState:
        """Prepare context based on protocol name only"""
        if state.get("error"):
            return state
            
        try:
            with get_db() as conn:
                # Get only protocol name
                protocol_query = text("""
                    SELECT name 
                    FROM medapp.protocols 
                    WHERE id = :protocol_id
                """)
                
                protocol_result = conn.execute(protocol_query, {
                    "protocol_id": state["protocol_id"]
                }).fetchone()
                
                if not protocol_result:
                    state["error"] = f"Protocol ID {state['protocol_id']} not found"
                    return state
                
                # Extract protocol name only
                protocol_name = protocol_result.name if hasattr(protocol_result, 'name') else "Unknown Protocol"
                
                # Analyze existing question types (for gap analysis)
                covered_types = []
                for q in state["existing_questions"]:
                    question_text = q["question"].lower()
                    if "how long" in question_text or "duration" in question_text:
                        covered_types.append("duration")
                    elif "how often" in question_text or "frequency" in question_text:
                        covered_types.append("frequency")
                    elif "severity" in question_text or "intense" in question_text or "rate" in question_text:
                        covered_types.append("severity")
                    elif "trigger" in question_text or "cause" in question_text:
                        covered_types.append("triggers")
                    elif "relief" in question_text or "help" in question_text:
                        covered_types.append("relief")
                
                covered_types = list(set(covered_types))
                
                # Simple context with just protocol name
                state["medical_context"] = {
                    "protocol_name": protocol_name,
                    "covered_question_types": covered_types,
                    "total_existing": len(state["existing_questions"])
                }
                
                state["metadata"]["protocol_name"] = protocol_name
                state["metadata"]["covered_types"] = covered_types
                
                logger.info(f"Protocol: {protocol_name}")
                logger.info(f"Covered question types: {covered_types}")
                logger.info(f"Total existing questions: {len(state['existing_questions'])}")
                
        except Exception as e:
            state["error"] = f"Error preparing protocol context: {e}"
            
        return state
    
    def _generate_with_llm(self, state: QuestionGenerationState) -> QuestionGenerationState:
        """Generate questions using LLM with protocol context"""
        if state.get("error"):
            return state
            
        try:
            logger.info("Generating questions with protocol-specific context...")
            
            response = self.llm_service.generate_questions(
                state["existing_questions"],
                state["medical_context"], 
                state["num_questions"]
            )
            
            state["generated_response"] = response
            state["metadata"]["llm_response_length"] = len(response)
            
            logger.info(f"LLM generated response of {len(response)} characters")
            logger.info(f"Response preview: {response[:200]}...")
            
        except Exception as e:
            state["error"] = f"Error generating with LLM: {e}"
            
        return state
    
    def _parse_response(self, state: QuestionGenerationState) -> QuestionGenerationState:
        """Parse LLM response into Question objects with enhanced fallback"""
        if state.get("error"):
            return state
            
        try:
            # Primary parsing attempt
            parsed_questions = self.parser.parse_llm_response(
                state["generated_response"], 
                state["next_q_id"]
            )
            
            # If primary parsing failed, try multiple fallback methods (like reference code)
            if not parsed_questions:
                logger.warning("Primary parsing failed, trying fallback methods...")
                
                # Method 1: Try reference code style parsing
                parsed_questions = self._fallback_parse_reference_style(
                    state["generated_response"], 
                    state["next_q_id"]
                )
                
                # Method 2: If still failed, try simple line-by-line parsing
                if not parsed_questions:
                    logger.warning("Reference style parsing failed, trying simple parsing...")
                    parsed_questions = self._fallback_parse_simple(
                        state["generated_response"], 
                        state["next_q_id"]
                    )
            
            if not parsed_questions:
                state["error"] = "No valid questions could be parsed from LLM response. Response format may be incorrect."
                logger.error(f"All parsing methods failed. LLM Response was: {state['generated_response']}")
                return state
            
            state["parsed_questions"] = parsed_questions
            state["metadata"]["parsed_questions_count"] = len(parsed_questions)
            
            logger.info(f"Successfully parsed {len(parsed_questions)} questions")
            
        except Exception as e:
            state["error"] = f"Error parsing response: {e}"
            
        return state
    
    def _fallback_parse_reference_style(self, response_text: str, start_q_id: int) -> List[Question]:
        """Fallback parsing using reference code style"""
        try:
            logger.info("Trying reference code style parsing...")
            questions = []
            
            # Split by various patterns
            lines = response_text.split('\n')
            current_question = None
            current_options = []
            option_id = 1
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this is a question line
                if ('question' in line.lower() and ('?' in line or ':' in line)) or line.endswith('?'):
                    # Save previous question if exists
                    if current_question and len(current_options) >= 2:
                        options = [Option(option_id=i+1, opt_value=opt["opt_value"]) 
                                 for i, opt in enumerate(current_options)]
                        questions.append(Question(
                            q_id=start_q_id + len(questions),
                            question=current_question,
                            options=options,
                            age_group="All",
                            gender="Both"
                        ))
                    
                    # Start new question
                    # Clean question text
                    current_question = re.sub(r'^\*\*.*?\*\*', '', line)  # Remove markdown
                    current_question = re.sub(r'^Question\s+\d+:\s*', '', current_question)  # Remove Question X:
                    current_question = re.sub(r'^\d+\.\s*', '', current_question)  # Remove number
                    current_question = current_question.strip()
                    if not current_question.endswith('?'):
                        current_question += '?'
                    current_options = []
                    option_id = 1
                
                # Check if this is an option line
                elif re.match(r'^\s*[1-4][\.\)]\s*', line) or re.match(r'^\s*[A-D][\.\)]\s*', line):
                    option_text = re.sub(r'^\s*[1-4A-D][\.\)]\s*', '', line).strip()
                    if option_text:
                        current_options.append({
                            "option_id": option_id,
                            "opt_value": option_text
                        })
                        option_id += 1
            
            # Save last question
            if current_question and len(current_options) >= 2:
                options = [Option(option_id=i+1, opt_value=opt["opt_value"]) 
                         for i, opt in enumerate(current_options)]
                questions.append(Question(
                    q_id=start_q_id + len(questions),
                    question=current_question,
                    options=options,
                    age_group="All",
                    gender="Both"
                ))
            
            logger.info(f"Reference style parsing extracted {len(questions)} questions")
            return questions
            
        except Exception as e:
            logger.error(f"Reference style parsing failed: {e}")
            return []
    
    def _fallback_parse_simple(self, response_text: str, start_q_id: int) -> List[Question]:
        """Simple fallback parsing - very basic"""
        try:
            logger.info("Trying simple fallback parsing...")
            questions = []
            
            # Find all question marks and try to extract questions
            sentences = response_text.split('?')
            
            for i, sentence in enumerate(sentences[:-1]):  # Exclude last empty part
                sentence = sentence.strip()
                if len(sentence) < 10:  # Too short to be a question
                    continue
                
                # Look for the question part
                question_text = sentence.split('\n')[-1].strip()  # Get last line before ?
                question_text = re.sub(r'^\*\*.*?\*\*', '', question_text)  # Remove markdown
                question_text = re.sub(r'^Question\s+\d+:\s*', '', question_text)  # Remove Question X:
                question_text = question_text.strip() + '?'
                
                # Create default options
                default_options = [
                    Option(option_id=1, opt_value="Yes"),
                    Option(option_id=2, opt_value="No"),
                    Option(option_id=3, opt_value="Sometimes")
                    
                ]
                
                questions.append(Question(
                    q_id=start_q_id + len(questions),
                    question=question_text,
                    options=default_options,
                    age_group="All",
                    gender="Both"
                ))
                
                # Limit to requested number
                if len(questions) >= 5:  # Don't create too many default questions
                    break
            
            logger.info(f"Simple parsing extracted {len(questions)} questions with default options")
            return questions
            
        except Exception as e:
            logger.error(f"Simple parsing failed: {e}")
            return []
    
    def _store_questions(self, state: QuestionGenerationState) -> QuestionGenerationState:
        """Store questions in database"""
        if state.get("error"):
            return state
            
        try:
            with get_db() as conn:
                stored_questions = []
                
                for q in state["parsed_questions"]:
                    # Insert question
                    insert_question_query = text("""
                        INSERT INTO medapp.questions (q_id, q_tag, qtn_doc_id, protocol_id, age_group, gender, question)
                        VALUES (:q_id, :q_tag, :qtn_doc_id, :protocol_id, :age_group, :gender, :question)
                    """)
                    
                    conn.execute(insert_question_query, {
                        "q_id": q.q_id,
                        "q_tag": "ai_generated",
                        "qtn_doc_id": state["doctor_id"],
                        "protocol_id": state["protocol_id"],
                        "age_group": q.age_group,
                        "gender": q.gender,
                        "question": q.question
                    })
                    
                    # Insert options
                    for option in q.options:
                        insert_option_query = text("""
                            INSERT INTO medapp.options (protocol_id, opt_doc_id, opt_q_id, option_value)
                            VALUES (:protocol_id, :opt_doc_id, :opt_q_id, :option_value)
                        """)
                        
                        conn.execute(insert_option_query, {
                            "protocol_id": state["protocol_id"],
                            "opt_doc_id": state["doctor_id"],
                            "opt_q_id": q.q_id,
                            "option_value": option.opt_value
                        })
                    
                    # Store for response
                    stored_questions.append({
                        "q_id": q.q_id,
                        "question": q.question,
                        "options": [{"option_id": opt.option_id, "opt_value": opt.opt_value} for opt in q.options],
                        "q_tag": "ai_generated",
                        "age_group": q.age_group,
                        "gender": q.gender
                    })
                    
                    logger.info(f"Stored question {q.q_id}: {q.question}")
                
                state["stored_questions"] = stored_questions
                state["metadata"]["stored_questions_count"] = len(stored_questions)
                
                logger.info(f"Successfully stored {len(stored_questions)} questions")
                
        except Exception as e:
            state["error"] = f"Error storing questions: {e}"
            
        return state
    
    def generate_additional_questions(self, protocol_id: int, doctor_id: int, num_questions: int) -> Dict[str, Any]:
        """Main method to generate additional questions"""
        # Initialize state
        initial_state = QuestionGenerationState(
            protocol_id=protocol_id,
            doctor_id=doctor_id,
            num_questions=num_questions,
            existing_questions=[],
            medical_context="",
            llm_prompt="",
            generated_response="",
            parsed_questions=[],
            stored_questions=[],
            next_q_id=1,
            error=None,
            metadata={}
        )
        
        # Run workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Prepare result
        result = {
            "status": "error" if final_state.get("error") else "success",
            "protocol_id": protocol_id,
            "doctor_id": doctor_id,
            "metadata": final_state["metadata"],
            "questions": final_state.get("stored_questions", [])
        }
        
        if final_state.get("error"):
            result["error"] = final_state["error"]
        
        return result

# ===================== API ENDPOINTS =====================

router = APIRouter(prefix='', tags=['Additional Question Generation'])

# Global workflow instance
_workflow_instance = None

def get_workflow():
    """Get or create workflow instance"""
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = AdditionalQuestionWorkflow()
    return _workflow_instance

@router.post("/GenerateAdditionalQuestions", response_model=List[Question])
async def generate_additional_questions(
    protocol_id: int = Query(..., description="Protocol ID to generate questions for"),
    doctor_id: int = Query(..., description="Doctor ID to generate questions for"),
    num_questions: int = Query(5, description="Number of additional questions to generate", ge=1, le=20)
):
    """
    Generate additional questions based on existing questions using Ollama Llama 3.1 8B with LangChain workflow
    """
    
    try:
        logger.info(f"🤖 Starting enhanced question generation for protocol_id={protocol_id}, doctor_id={doctor_id}")
        
        # Get workflow instance
        workflow = get_workflow()
        
        # Generate questions
        result = workflow.generate_additional_questions(protocol_id, doctor_id, num_questions)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
        
        logger.info(f"🎉 Successfully generated {len(result['questions'])} questions")
        
        return JSONResponse(content=result["questions"])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🚨 Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/GetExistingQuestions")
async def get_existing_questions(
    protocol_id: int = Query(..., description="Protocol ID"),
    doctor_id: int = Query(..., description="Doctor ID")
):
    """
    Get existing questions for a protocol and doctor with enhanced analysis
    """
    try:
        with get_db() as conn:
            # Get existing questions
            existing_questions_query = text("""
                SELECT q.q_id, q.question, q.age_group, q.gender, q.q_tag
                FROM medapp.questions q 
                WHERE q.protocol_id = :protocol_id AND q.qtn_doc_id = :doctor_id
                ORDER BY q.q_id
            """)
            
            questions_result = conn.execute(existing_questions_query, {
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
        logger.error(f"Error retrieving existing questions: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving questions: {str(e)}")