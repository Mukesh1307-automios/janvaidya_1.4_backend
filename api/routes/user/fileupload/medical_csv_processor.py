import os
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, TypedDict
from pathlib import Path
import json
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== WORKFLOW STATE =====================

class CSVWorkflowState(TypedDict):
    file_path: str
    file_type: str  # 'csv', 'xlsx', 'xls'
    user_prompt: str
    dataframe: Optional[pd.DataFrame]
    question_rows: List[Dict[str, Any]]
    formatted_response: str
    error: Optional[str]
    metadata: Dict[str, Any]

# ===================== SIMPLE CSV PROCESSOR =====================

class SimpleCSVProcessor:
    """Basic CSV/Excel processor that extracts all questions and options without filtering"""

    def load_file(self, file_path: str, file_type: str) -> pd.DataFrame:
        """Load CSV or Excel file into DataFrame"""
        try:
            if file_type == 'csv':
                # Try different encodings and separators
                for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        # Try different separators
                        for sep in [',', ';', '\t', '|']:
                            try:
                                df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                                if len(df.columns) > 1:  # Ensure we have multiple columns
                                    logger.info(f"Successfully loaded CSV with encoding={encoding}, separator='{sep}'")
                                    return df
                            except:
                                continue
                    except UnicodeDecodeError:
                        continue
                
                # Last resort - try with default settings
                return pd.read_csv(file_path)
            
            elif file_type in ['xlsx', 'xls']:
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise ValueError(f"Could not load file: {str(e)}")

    def extract_all_questions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract all questions and options from DataFrame"""
        question_rows = []
        
        try:
            # Clean column names - remove leading/trailing spaces
            df.columns = df.columns.str.strip()
            
            logger.info(f"DataFrame columns: {list(df.columns)}")
            logger.info(f"DataFrame shape: {df.shape}")
            
            for index, row in df.iterrows():
                question = ""
                options = {}
                
                # Find question column - more flexible matching
                for col in df.columns:
                    col_lower = col.lower().strip()
                    val = str(row[col]).strip() if pd.notna(row[col]) else ""
                    
                    # Check if this column contains question text
                    if val and val != 'nan' and val.lower() != 'none':
                        # Look for question-like patterns in column name or content
                        is_question_col = any(keyword in col_lower for keyword in [
                            'question', 'text', 'query', 'prompt', 'q_', 'quest'
                        ])
                        
                        # Also check if the content looks like a question
                        is_question_content = ('?' in val or 
                                             any(word in val.lower() for word in [
                                                 'what', 'how', 'when', 'where', 'why', 
                                                 'which', 'who', 'do you', 'have you', 'are you'
                                             ]))
                        
                        if is_question_col or (is_question_content and not question):
                            question = val
                            break
                
                # Find option columns - more flexible matching
                for col in df.columns:
                    col_lower = col.lower().strip()
                    val = str(row[col]).strip() if pd.notna(row[col]) else ""
                    
                    if val and val != 'nan' and val.lower() != 'none' and val != question:
                        # Check if this column contains option text
                        is_option_col = any(keyword in col_lower for keyword in [
                            'option', 'choice', 'answer', 'opt', 'choice'
                        ])
                        
                        # Check if column name looks like an option (A, B, C, D, etc.)
                        is_option_letter = (col_lower in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] or
                                          col_lower.startswith(('a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)')) or
                                          col_lower.startswith(('option_', 'opt_', 'choice_')))
                        
                        # Check if it's a numbered option column
                        is_numbered_option = col_lower.startswith(('1', '2', '3', '4', '5', '6', '7', '8')) and len(col_lower) <= 3
                        
                        if is_option_col or is_option_letter or is_numbered_option:
                            options[col] = val

                # If no explicit question found, try to find the longest text field
                if not question and options:
                    longest_text = ""
                    for col in df.columns:
                        val = str(row[col]).strip() if pd.notna(row[col]) else ""
                        if val and len(val) > len(longest_text) and val not in options.values():
                            longest_text = val
                    if longest_text:
                        question = longest_text

                # Add to results if we have both question and options
                if question and options and len(options) >= 2:
                    question_rows.append({
                        "question": question, 
                        "options": options,
                        "row_index": index + 1
                    })
                    logger.debug(f"Row {index + 1}: Found question with {len(options)} options")
            
            logger.info(f"Extracted {len(question_rows)} questions from CSV")
            return question_rows
            
        except Exception as e:
            logger.error(f"Error extracting questions: {e}")
            return []

    def format_questions(self, question_rows: List[Dict[str, Any]]) -> str:
        """Format questions for display with consistent lettered options"""
        if not question_rows:
            return "No questions found in the CSV file. Please ensure your CSV has a question column and option columns (A, B, C, D or Option_A, Option_B, etc.)."

        formatted_parts = []
        option_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        
        for i, q in enumerate(question_rows, 1):
            # Format question
            question_text = q['question'].strip()
            if not question_text.endswith('?'):
                question_text += '?'
            
            part = f"Question {i}: {question_text}\n"
            
            # Format options with consistent lettering
            option_items = list(q['options'].items())
            for j, (col, opt) in enumerate(option_items):
                if j < len(option_letters):
                    part += f"   {option_letters[j]}) {opt}\n"
                else:
                    part += f"   {j+1}) {opt}\n"
            
            formatted_parts.append(part)

        return "\n".join(formatted_parts)

    def validate_csv_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate CSV structure and provide feedback"""
        validation_result = {
            "is_valid": False,
            "has_questions": False,
            "has_options": False,
            "question_columns": [],
            "option_columns": [],
            "issues": [],
            "suggestions": [],
            "total_rows": len(df),
            "total_columns": len(df.columns)
        }
        
        try:
            # Clean column names for analysis
            df.columns = df.columns.str.strip()
            columns_lower = [col.lower() for col in df.columns]
            
            # Check for question columns
            question_keywords = ['question', 'text', 'query', 'prompt', 'q_', 'quest']
            question_columns = []
            
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in question_keywords):
                    question_columns.append(col)
            
            validation_result["question_columns"] = question_columns
            validation_result["has_questions"] = len(question_columns) > 0
            
            # Check for option columns
            option_keywords = ['option', 'choice', 'answer', 'opt']
            option_columns = []
            
            for col in df.columns:
                col_lower = col.lower()
                if (any(keyword in col_lower for keyword in option_keywords) or 
                    col_lower in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] or
                    col_lower.startswith(('a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)')) or
                    col_lower.startswith(('option_', 'opt_', 'choice_'))):
                    option_columns.append(col)
            
            validation_result["option_columns"] = option_columns
            validation_result["has_options"] = len(option_columns) >= 2
            
            # Overall validation
            validation_result["is_valid"] = validation_result["has_questions"] and validation_result["has_options"]
            
            # Generate issues and suggestions
            if not validation_result["has_questions"]:
                validation_result["issues"].append("No question column found")
                validation_result["suggestions"].append("Add a column named 'Question', 'Question_Text', or 'Text'")
            
            if not validation_result["has_options"]:
                validation_result["issues"].append(f"Insufficient option columns found (found {len(option_columns)}, need at least 2)")
                validation_result["suggestions"].append("Add columns named 'Option_A', 'Option_B', 'Option_C', 'Option_D' or simply 'A', 'B', 'C', 'D'")
            
            if len(option_columns) < 2:
                validation_result["suggestions"].append("Add more option columns for proper question formatting")
            
            # Check data quality
            empty_rows = df.isnull().all(axis=1).sum()
            if empty_rows > 0:
                validation_result["issues"].append(f"Found {empty_rows} completely empty rows")
                validation_result["suggestions"].append("Remove empty rows from your CSV")
            
            return validation_result
            
        except Exception as e:
            validation_result["issues"].append(f"Error during validation: {str(e)}")
            return validation_result

    def get_csv_preview(self, df: pd.DataFrame, max_rows: int = 5) -> Dict[str, Any]:
        """Get a preview of the CSV structure"""
        try:
            preview_data = df.head(max_rows).fillna("").to_dict('records')
            
            return {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "columns": list(df.columns),
                "preview_data": preview_data,
                "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "has_null_values": df.isnull().any().any(),
                "null_counts": df.isnull().sum().to_dict()
            }
        except Exception as e:
            logger.error(f"Error creating CSV preview: {e}")
            return {"error": str(e)}

# ===================== ENHANCED CSV PROCESSOR =====================

class EnhancedCSVProcessor(SimpleCSVProcessor):
    """Enhanced CSV processor with additional features"""
    
    def __init__(self):
        super().__init__()
        self.supported_question_patterns = [
            r'question',
            r'text',
            r'query',
            r'prompt',
            r'q\d*',
        ]
        
        self.supported_option_patterns = [
            r'option[_\s]*[a-z]',
            r'choice[_\s]*[a-z]',
            r'answer[_\s]*[a-z]',
            r'^[a-f]$',
            r'opt[_\s]*[a-z]',
        ]
    
    def smart_column_detection(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Smart detection of question and option columns"""
        columns = {
            "question_columns": [],
            "option_columns": [],
            "other_columns": [],
            "potential_question_columns": [],
            "potential_option_columns": []
        }
        
        for col in df.columns:
            col_clean = col.lower().strip()
            
            # Check if it's a question column
            is_question = any(pattern in col_clean for pattern in ['question', 'text', 'query', 'prompt'])
            
            # Check if it's an option column
            is_option = (any(pattern in col_clean for pattern in ['option', 'choice', 'answer']) or
                        col_clean in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] or
                        col_clean.startswith(('a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)')) or
                        col_clean.startswith(('option_', 'opt_', 'choice_')))
            
            # Check for potential columns based on content
            sample_values = df[col].dropna().head(3).astype(str).tolist()
            
            potential_question = any('?' in val or len(val) > 20 for val in sample_values)
            potential_option = all(len(val) < 50 and val.strip() for val in sample_values) and len(sample_values) > 0
            
            if is_question:
                columns["question_columns"].append(col)
            elif is_option:
                columns["option_columns"].append(col)
            elif potential_question and not is_option:
                columns["potential_question_columns"].append(col)
            elif potential_option and not is_question:
                columns["potential_option_columns"].append(col)
            else:
                columns["other_columns"].append(col)
        
        return columns
    
    def extract_questions_with_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract questions with validation and error reporting"""
        result = {
            "success": False,
            "questions": [],
            "errors": [],
            "warnings": [],
            "metadata": {}
        }
        
        try:
            # Validate structure first
            validation = self.validate_csv_structure(df)
            result["metadata"]["validation"] = validation
            
            if not validation["is_valid"]:
                result["errors"].extend(validation["issues"])
                # Try to extract anyway if we have some data
                questions = self.extract_all_questions(df)
                if questions:
                    result["warnings"].append("CSV structure is not ideal, but questions were found")
                    result["questions"] = questions
                    result["success"] = True
                return result
            
            # Extract questions
            questions = self.extract_all_questions(df)
            
            if not questions:
                result["errors"].append("No valid questions found in the CSV")
                result["errors"].append("Please check that your CSV has:")
                result["errors"].append("1. A column with questions (named 'Question' or containing '?')")
                result["errors"].append("2. Columns with options (named 'Option_A', 'A', etc.)")
                return result
            
            # Validate each question
            validated_questions = []
            for i, q in enumerate(questions):
                if len(q["options"]) < 2:
                    result["warnings"].append(f"Question {i+1} has less than 2 options")
                elif len(q["options"]) > 8:
                    result["warnings"].append(f"Question {i+1} has more than 8 options (truncated)")
                    # Keep only first 8 options
                    limited_options = dict(list(q["options"].items())[:8])
                    q["options"] = limited_options
                    validated_questions.append(q)
                else:
                    validated_questions.append(q)
            
            result["questions"] = validated_questions
            result["success"] = True
            result["metadata"]["total_questions"] = len(validated_questions)
            result["metadata"]["skipped_questions"] = len(questions) - len(validated_questions)
            
            return result
            
        except Exception as e:
            result["errors"].append(f"Processing error: {str(e)}")
            logger.error(f"Error in extract_questions_with_validation: {e}")
            return result

    def auto_fix_csv_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attempt to automatically fix common CSV structure issues"""
        try:
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            # Remove completely empty columns
            df = df.dropna(axis=1, how='all')
            
            # Clean column names
            df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace('\r', '')
            
            # Fill NaN values with empty strings for text processing
            df = df.fillna('')
            
            logger.info(f"Auto-fixed CSV structure: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error in auto_fix_csv_structure: {e}")
            return df

# ===================== MAIN CSV INTERFACE =====================

class MedicalCSVQuestionProcessor:
    """Main interface for CSV-based question processing"""
    
    def __init__(self):
        self.processor = EnhancedCSVProcessor()
    
    def process(self, file_path: str, prompt: str = "") -> Dict[str, Any]:
        """Main processing method"""
        try:
            logger.info(f"Processing CSV file: {file_path}")
            
            # Determine file type
            ext = Path(file_path).suffix.lower().replace('.', '')
            file_type = ext if ext in ['csv', 'xlsx', 'xls'] else None
            
            if not file_type:
                return {
                    "status": "error",
                    "error": f"Unsupported file type: {ext}. Supported types: csv, xlsx, xls",
                    "metadata": {}
                }
            
            # Load file
            try:
                df = self.processor.load_file(file_path, file_type)
                logger.info(f"Loaded file with shape: {df.shape}")
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"Failed to load file: {str(e)}",
                    "metadata": {"file_type": file_type}
                }
            
            # Auto-fix common issues
            df = self.processor.auto_fix_csv_structure(df)
            
            # Extract questions with validation
            extraction_result = self.processor.extract_questions_with_validation(df)
            
            if not extraction_result["success"]:
                return {
                    "status": "error", 
                    "error": " | ".join(extraction_result["errors"]),
                    "warnings": extraction_result.get("warnings", []),
                    "metadata": extraction_result.get("metadata", {})
                }
            
            # Format questions
            formatted_response = self.processor.format_questions(extraction_result["questions"])
            
            return {
                "status": "success",
                "response": formatted_response,
                "metadata": {
                    "total_questions": len(extraction_result["questions"]),
                    "file_type": file_type,
                    "columns": list(df.columns),
                    "warnings": extraction_result.get("warnings", []),
                    "validation": extraction_result.get("metadata", {}).get("validation", {}),
                    "file_shape": df.shape
                }
            }
            
        except Exception as e:
            logger.error(f"Error in CSV processing: {e}")
            return {
                "status": "error", 
                "error": f"Unexpected error: {str(e)}",
                "metadata": {}
            }
    
    def preview_file(self, file_path: str) -> Dict[str, Any]:
        """Preview CSV file structure"""
        try:
            ext = Path(file_path).suffix.lower().replace('.', '')
            file_type = ext if ext in ['csv', 'xlsx', 'xls'] else None
            
            if not file_type:
                return {"error": f"Unsupported file type: {ext}"}
            
            df = self.processor.load_file(file_path, file_type)
            df = self.processor.auto_fix_csv_structure(df)
            
            # Get preview and validation
            preview = self.processor.get_csv_preview(df)
            validation = self.processor.validate_csv_structure(df)
            column_detection = self.processor.smart_column_detection(df)
            
            return {
                "preview": preview,
                "validation": validation,
                "column_detection": column_detection,
                "file_type": file_type,
                "recommendations": self._get_recommendations(validation, column_detection)
            }
            
        except Exception as e:
            logger.error(f"Error previewing file: {e}")
            return {"error": str(e)}
    
    def _get_recommendations(self, validation: Dict, column_detection: Dict) -> List[str]:
        """Generate recommendations for improving CSV structure"""
        recommendations = []
        
        if not validation["has_questions"]:
            if column_detection["potential_question_columns"]:
                recommendations.append(f"Consider renaming '{column_detection['potential_question_columns'][0]}' to 'Question' or 'Question_Text'")
            else:
                recommendations.append("Add a column named 'Question' or 'Question_Text' containing your questions")
        
        if not validation["has_options"]:
            if column_detection["potential_option_columns"]:
                recommendations.append("Consider renaming option columns to 'Option_A', 'Option_B', 'Option_C', 'Option_D'")
            else:
                recommendations.append("Add columns named 'Option_A', 'Option_B', 'Option_C', 'Option_D' for answer choices")
        
        if validation["total_rows"] == 0:
            recommendations.append("Your CSV appears to be empty")
        
        if len(validation["option_columns"]) < 4:
            recommendations.append("Consider adding more option columns (A, B, C, D) for better question formatting")
        
        return recommendations
    
    def get_format_requirements(self) -> Dict[str, Any]:
        """Get CSV format requirements"""
        return {
            "required_columns": {
                "question_column": {
                    "names": ["Question", "Question_Text", "Text", "Query"],
                    "description": "Column containing the question text"
                },
                "option_columns": {
                    "names": ["Option_A", "Option_B", "Option_C", "Option_D", "A", "B", "C", "D"],
                    "description": "Columns containing the answer options",
                    "minimum": 2
                }
            },
            "example_format": {
                "headers": ["Question_Text", "Option_A", "Option_B", "Option_C", "Option_D"],
                "sample_rows": [
                    ["Do you have difficulty breathing?", "Yes at rest", "Yes with activity", "Yes when lying down", "No breathing normal"],
                    ["How often do you exercise?", "Daily", "Weekly", "Monthly", "Never"],
                    ["What is your pain level?", "Severe", "Moderate", "Mild", "None"]
                ]
            },
            "supported_file_types": ["csv", "xlsx", "xls"],
            "encoding_support": ["utf-8", "latin-1", "cp1252", "iso-8859-1"],
            "separator_support": [",", ";", "\t", "|"]
        }

# ===================== CSV VALIDATION UTILITIES =====================

def validate_csv_file(file_path: str) -> Dict[str, Any]:
    """Standalone function to validate CSV file"""
    processor = MedicalCSVQuestionProcessor()
    return processor.preview_file(file_path)

def convert_csv_format(input_file: str, output_file: str, format_type: str = "standard") -> Dict[str, Any]:
    """Convert CSV to standard format"""
    try:
        processor = MedicalCSVQuestionProcessor()
        
        # Process the file
        result = processor.process(input_file)
        
        if result["status"] != "success":
            return {"error": result.get("error", "Processing failed")}
        
        # Save formatted output
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result["response"])
        
        return {
            "success": True,
            "output_file": output_file,
            "questions_processed": result["metadata"]["total_questions"]
        }
        
    except Exception as e:
        return {"error": str(e)}

def fix_csv_format(input_file: str, output_file: str) -> Dict[str, Any]:
    """Attempt to fix CSV format issues"""
    try:
        processor = MedicalCSVQuestionProcessor()
        
        # Load and auto-fix the file
        ext = Path(input_file).suffix.lower().replace('.', '')
        file_type = ext if ext in ['csv', 'xlsx', 'xls'] else None
        
        if not file_type:
            return {"error": f"Unsupported file type: {ext}"}
        
        df = processor.processor.load_file(input_file, file_type)
        df_fixed = processor.processor.auto_fix_csv_structure(df)
        
        # Save the fixed file
        if file_type == 'csv':
            df_fixed.to_csv(output_file, index=False, encoding='utf-8')
        else:
            df_fixed.to_excel(output_file, index=False)
        
        return {
            "success": True,
            "output_file": output_file,
            "original_shape": df.shape,
            "fixed_shape": df_fixed.shape
        }
        
    except Exception as e:
        return {"error": str(e)}

# ===================== CLI INTERFACE =====================

def main():
    """CLI interface for testing CSV processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CSV Question Processor")
    parser.add_argument("--file", required=True, help="Path to CSV/Excel file")
    parser.add_argument("--prompt", default="Extract all questions", help="Processing prompt")
    parser.add_argument("--preview", action="store_true", help="Preview file structure only")
    parser.add_argument("--validate", action="store_true", help="Validate file format only")
    parser.add_argument("--output", help="Save output to file")
    parser.add_argument("--format", action="store_true", help="Show format requirements")
    parser.add_argument("--fix", help="Fix CSV format and save to specified file")
    parser.add_argument("--convert", help="Convert to standard format and save to specified file")
    
    args = parser.parse_args()
    
    processor = MedicalCSVQuestionProcessor()
    
    if args.format:
        requirements = processor.get_format_requirements()
        print("CSV Format Requirements:")
        print(json.dumps(requirements, indent=2))
        return
    
    if args.fix:
        result = fix_csv_format(args.file, args.fix)
        if result.get("success"):
            print(f"Fixed CSV saved to: {args.fix}")
            print(f"Original shape: {result['original_shape']}")
            print(f"Fixed shape: {result['fixed_shape']}")
        else:
            print(f"Error fixing CSV: {result.get('error')}")
        return
    
    if args.convert:
        result = convert_csv_format(args.file, args.convert)
        if result.get("success"):
            print(f"Converted file saved to: {args.convert}")
            print(f"Questions processed: {result['questions_processed']}")
        else:
            print(f"Error converting: {result.get('error')}")
        return
    
    if args.validate or args.preview:
        preview = processor.preview_file(args.file)
        print("File Preview and Validation:")
        print(json.dumps(preview, indent=2))
        return
    
    # Process the file
    print(f"Processing file: {args.file}")
    result = processor.process(args.file, args.prompt)
    
    if result["status"] == "success":
        print("\n" + "="*50)
        print("EXTRACTED QUESTIONS:")
        print("="*50)
        print(result["response"])
        print("="*50)
        print(f"Total questions: {result['metadata']['total_questions']}")
        print(f"File shape: {result['metadata']['file_shape']}")
        
        if result["metadata"].get("warnings"):
            print("\nWarnings:")
            for warning in result["metadata"]["warnings"]:
                print(f"  - {warning}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        if result.get("warnings"):
            print("Warnings:")
            for warning in result["warnings"]:
                print(f"  - {warning}")
    
    # Save output if requested
    if args.output and result["status"] == "success":
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result["response"])
        print(f"\nOutput saved to: {args.output}")

if __name__ == "__main__":
    main()