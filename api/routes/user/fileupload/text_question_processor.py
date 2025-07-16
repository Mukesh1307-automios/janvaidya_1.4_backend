import re
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Setup logging
logger = logging.getLogger(__name__)

# ===================== PYDANTIC MODELS =====================

class Option(BaseModel):
    option_id: int
    opt_value: str

class Question(BaseModel):
    q_id: int
    question: str
    options: List[Option]
    age_group: str = "All"
    gender: str = "Both"

# ===================== UNIVERSAL QUESTION PARSER =====================

class UniversalQuestionParser:
    """
    Production-ready parser that handles 5 different question formats:
    
    Format 1: Standard Lettered (A), B), C), D))
    Format 2: Numbered (1), 2), 3), 4))  
    Format 3: Single Line Space-Separated
    Format 4: Multi-line Plain Text
    Format 5: Mixed Delimiters (commas, semicolons, pipes)
    """
    
    def __init__(self):
        self.supported_formats = {
            "lettered": r"[A-Z]\)",
            "numbered": r"\d+\)",
            "lettered_dots": r"[A-Z]\.",
            "numbered_dots": r"\d+\.",
            "parenthetical": r"\([A-Z]\)",
            "lowercase": r"[a-z]\)",
            "bullets": r"[-\*\•]",  # FIXED: moved - to beginning
            "space_separated": r"\s{2,}",
            "comma_separated": r",",
            "semicolon_separated": r";",
            "pipe_separated": r"\|"
        }
    
    def detect_format(self, text: str) -> str:
        """Auto-detect the format of input text"""
        formats_found = []
        
        # Check for explicit markers
        for format_name, pattern in self.supported_formats.items():
            if re.search(pattern, text):
                formats_found.append(format_name)
        
        # Check for questions
        has_questions = bool(re.search(r'\?|question\s*\d*[:.]|^\d+\.\s+.*(?:\?|$)', text, re.IGNORECASE | re.MULTILINE))
        
        if not formats_found and has_questions:
            return "implicit_options"
        elif formats_found:
            return f"explicit_markers_{'+'.join(formats_found[:2])}"
        else:
            return "unknown"
    
    def parse_questions(self, text: str) -> List[Question]:
        """
        Universal parser that tries multiple parsing strategies
        """
        text = text.strip()
        if not text:
            return []
        
        questions = []
        
        # Strategy 1: Try regex-based parsing for structured formats
        questions = self._parse_structured_formats(text)
        if questions:
            return questions
        
        # Strategy 2: Try line-by-line parsing
        questions = self._parse_line_by_line(text)
        if questions:
            return questions
        
        # Strategy 3: Try single-line parsing
        questions = self._parse_single_line_formats(text)
        if questions:
            return questions
        
        # Strategy 4: Try CSV-style parsing
        questions = self._parse_csv_style(text)
        if questions:
            return questions
        
        # Strategy 5: Last resort - intelligent splitting
        questions = self._parse_intelligent_split(text)
        
        return questions
    
    def _parse_structured_formats(self, text: str) -> List[Question]:
        """Parse Format 1 & 2: Structured with explicit markers"""
        questions = []
        
        # Pattern 1: "1. Question? A) option B) option C) option D) option"
        pattern1 = r'(\d+)\.\s*([^?]+\?)\s*A\)\s*([^B]+?)\s*B\)\s*([^C]+?)\s*C\)\s*([^D]+?)\s*D\)\s*([^1-9]*?)(?=\d+\.\s*[^?]*\?|$)'
        matches = re.findall(pattern1, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            question_text = match[1].strip()
            options = [match[2].strip(), match[3].strip(), match[4].strip(), match[5].strip()]
            options = [self._clean_option(opt) for opt in options if opt.strip()]
            
            if len(options) >= 2:
                questions.append(self._create_question(len(questions) + 1, question_text, options))
        
        if questions:
            return questions
        
        # Pattern 2: "Question 1: Text? A) option B) option C) option D) option"
        pattern2 = r'(?:Question\s*\d*[:\.]|Q\d*[:\.])\s*([^?]+\?)\s*A\)\s*([^B]+?)\s*B\)\s*([^C]+?)\s*C\)\s*([^D]+?)\s*D\)\s*([^Q]*?)(?=Question\s*\d*[:\.]|Q\d*[:\.]|$)'
        matches = re.findall(pattern2, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            question_text = match[0].strip()
            options = [match[1].strip(), match[2].strip(), match[3].strip(), match[4].strip()]
            options = [self._clean_option(opt) for opt in options if opt.strip()]
            
            if len(options) >= 2:
                questions.append(self._create_question(len(questions) + 1, question_text, options))
        
        if questions:
            return questions
        
        # Pattern 3: Numbered options "Question? 1) option 2) option 3) option 4) option"
        pattern3 = r'([^?]+\?)\s*1\)\s*([^2]+?)\s*2\)\s*([^3]+?)\s*3\)\s*([^4]+?)\s*4\)\s*([^1-9]*?)(?=[^?]*\?|$)'
        matches = re.findall(pattern3, text, re.DOTALL)
        
        for match in matches:
            question_text = match[0].strip()
            options = [match[1].strip(), match[2].strip(), match[3].strip(), match[4].strip()]
            options = [self._clean_option(opt) for opt in options if opt.strip()]
            
            if len(options) >= 2:
                questions.append(self._create_question(len(questions) + 1, question_text, options))
        
        return questions
    
    def _parse_line_by_line(self, text: str) -> List[Question]:
        """Parse Format 4: Multi-line format"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        questions = []
        current_question = None
        current_options = []
        
        for line in lines:
            # Check if line is a question
            if self._is_question_line(line):
                # Save previous question
                if current_question and current_options:
                    questions.append(self._create_question(len(questions) + 1, current_question, current_options))
                
                # Start new question
                current_question = self._clean_question_text(line)
                current_options = []
            
            # Check if line is an option
            elif self._is_option_line(line):
                if current_question:
                    option_text = self._extract_option_text(line)
                    if option_text:
                        current_options.append(option_text)
        
        # Add last question
        if current_question and current_options:
            questions.append(self._create_question(len(questions) + 1, current_question, current_options))
        
        return questions
    
    def _parse_single_line_formats(self, text: str) -> List[Question]:
        """Parse Format 3 & 5: Single line and mixed delimiters"""
        questions = []
        
        # Split into potential question blocks
        question_blocks = self._split_into_question_blocks(text)
        
        for block in question_blocks:
            parsed_question = self._parse_single_question_block(block)
            if parsed_question:
                questions.append(self._create_question(len(questions) + 1, 
                                                     parsed_question['question'], 
                                                     parsed_question['options']))
        
        return questions
    
    def _parse_csv_style(self, text: str) -> List[Question]:
        """Parse CSV-formatted questions"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        questions = []
        
        for line in lines:
            # Look for CSV-like patterns with commas or pipes
            if ',' in line or '|' in line:
                parts = re.split(r'[,|]', line)
                if len(parts) >= 3:  # At least question + 2 options
                    question_text = parts[0].strip()
                    options = [part.strip() for part in parts[1:] if part.strip()]
                    
                    if self._looks_like_question(question_text) and len(options) >= 2:
                        questions.append(self._create_question(len(questions) + 1, question_text, options))
        
        return questions
    
    def _parse_intelligent_split(self, text: str) -> List[Question]:
        """Last resort: Intelligent splitting based on patterns"""
        questions = []
        
        # Find all question-like sentences
        question_sentences = re.findall(r'[^.!]*\?[^.!]*', text)
        
        for sentence in question_sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Try to find the question part and options part
            question_match = re.search(r'^(.*?\?)', sentence)
            if question_match:
                question_text = question_match.group(1).strip()
                remaining_text = sentence[len(question_text):].strip()
                
                if remaining_text:
                    # Try different splitting methods
                    options = self._extract_options_from_text(remaining_text)
                    if len(options) >= 2:
                        questions.append(self._create_question(len(questions) + 1, question_text, options))
        
        return questions
    
    def _split_into_question_blocks(self, text: str) -> List[str]:
        """Split text into individual question blocks"""
        # Method 1: Split by numbered questions
        if re.search(r'\d+\.\s+', text):
            blocks = re.split(r'(?=\d+\.\s+)', text)
            return [block.strip() for block in blocks if block.strip()]
        
        # Method 2: Split by question patterns
        blocks = re.split(r'(?=[A-Z][^?]*\?)', text)
        return [block.strip() for block in blocks if block.strip() and '?' in block]
    
    def _parse_single_question_block(self, block: str) -> Optional[Dict[str, Any]]:
        """Parse a single question block"""
        # Find question part
        question_match = re.search(r'^[^?]*\?', block)
        if not question_match:
            return None
        
        question_text = question_match.group().strip()
        remaining_text = block[len(question_text):].strip()
        
        if not remaining_text:
            return None
        
        # Extract options from remaining text
        options = self._extract_options_from_text(remaining_text)
        
        if len(options) >= 2:
            return {
                'question': self._clean_question_text(question_text),
                'options': options
            }
        
        return None
    
    def _extract_options_from_text(self, text: str) -> List[str]:
        """Extract options from text using multiple methods"""
        options = []
        
        # Method 1: Explicit markers (A), B), 1), 2), etc.)
        for pattern in [r'[A-Z]\)', r'\d+\)', r'[A-Z]\.', r'\d+\.', r'\([A-Z]\)', r'\(\d+\)']:
            if re.search(pattern, text):
                parts = re.split(f'({pattern})', text)
                current_option = ""
                for part in parts:
                    if re.match(pattern, part):
                        if current_option.strip():
                            options.append(current_option.strip())
                        current_option = ""
                    else:
                        current_option += part
                if current_option.strip():
                    options.append(current_option.strip())
                break
        
        # Method 2: Multiple spaces (space-separated)
        if not options and '  ' in text:
            options = [opt.strip() for opt in re.split(r'\s{2,}', text) if opt.strip()]
        
        # Method 3: Comma-separated
        if not options and ',' in text:
            options = [opt.strip() for opt in text.split(',') if opt.strip()]
        
        # Method 4: Semicolon-separated
        if not options and ';' in text:
            options = [opt.strip() for opt in text.split(';') if opt.strip()]
        
        # Method 5: Pipe-separated
        if not options and '|' in text:
            options = [opt.strip() for opt in text.split('|') if opt.strip()]
        
        # Method 6: Word-based splitting for simple cases
        if not options:
            words = text.split()
            if len(words) >= 2 and len(words) <= 8:  # Reasonable number of words for options
                # Check if words look like options (not a sentence)
                if not any(word.lower() in ['the', 'and', 'or', 'but', 'with', 'from', 'that', 'this'] for word in words):
                    options = words
        
        return [self._clean_option(opt) for opt in options if opt and len(opt.strip()) > 0]
    
    def _is_question_line(self, line: str) -> bool:
        """Check if line is a question"""
        if not line:
            return False
        
        # Explicit question patterns
        question_patterns = [
            r'\?$',  # Ends with ?
            r'^\d+\.\s+.*\?',  # Numbered question
            r'^(?:Question|Q)\s*\d*[:\.]',  # Question label
            r'^(?:What|How|When|Where|Why|Which|Who|Do|Does|Is|Are|Can|Will|Should|Have)\b',  # Question words
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        
        return False
    
    def _is_option_line(self, line: str) -> bool:
        """Check if line is an option"""
        if not line:
            return False
        
        option_patterns = [
            r'^[A-Z]\)',  # A) B) C) D)
            r'^[A-Z]\.',  # A. B. C. D.
            r'^[a-z]\)',  # a) b) c) d)
            r'^[a-z]\.',  # a. b. c. d.
            r'^\d+\)',    # 1) 2) 3) 4)
            r'^\d+\.',    # 1. 2. 3. 4.
            r'^\([A-Z]\)',  # (A) (B) (C) (D)
            r'^\(\d+\)',    # (1) (2) (3) (4)
            r'^[-\*\•]\s+', # FIXED: moved - to beginning, added \s+
        ]
        
        for pattern in option_patterns:
            if re.match(pattern, line):
                return True
        
        return False
    
    def _extract_option_text(self, line: str) -> str:
        """Extract option text from line"""
        # FIXED: Updated regex patterns to avoid character range issues
        # Remove option markers
        patterns_to_remove = [
            r'^[A-Z]\)\s*',     # A) 
            r'^[A-Z]\.\s*',     # A.
            r'^[a-z]\)\s*',     # a)
            r'^[a-z]\.\s*',     # a.
            r'^\d+\)\s*',       # 1)
            r'^\d+\.\s*',       # 1.
            r'^\([A-Z]\)\s*',   # (A)
            r'^\(\d+\)\s*',     # (1)
            r'^[-]\s*',         # -
            r'^[\*]\s*',        # *
            r'^[\•]\s*',        # •
        ]
        
        cleaned = line
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, count=1)
            if cleaned != line:  # If we found and removed a pattern, stop
                break
                
        return cleaned.strip()
    
    def _clean_question_text(self, text: str) -> str:
        """Clean question text"""
        # Remove question numbering
        text = re.sub(r'^\d+\.\s*', '', text)
        text = re.sub(r'^(?:Question|Q)\s*\d*[:\.\s]*', '', text, flags=re.IGNORECASE)
        return text.strip()
    
    def _clean_option(self, text: str) -> str:
        """Clean option text"""
        if not text:
            return ""
        
        # Remove common prefixes - FIXED regex patterns
        patterns_to_remove = [
            r'^[A-Z]\)\s*',     # A) 
            r'^[A-Z]\.\s*',     # A.
            r'^[a-z]\)\s*',     # a)
            r'^[a-z]\.\s*',     # a.
            r'^\d+\)\s*',       # 1)
            r'^\d+\.\s*',       # 1.
            r'^\([A-Z]\)\s*',   # (A)
            r'^\(\d+\)\s*',     # (1)
            r'^[-]\s*',         # -
            r'^[\*]\s*',        # *
            r'^[\•]\s*',        # •
            r'^[:\-\*\•]+\s*',  # Multiple symbols
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, count=1)
        
        # Remove trailing question fragments
        text = re.sub(r'\s+\d+\.\s*.*$', '', text)
        text = re.sub(r'\s+(?:Question|Q)\s*\d*[:\.].*$', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _looks_like_question(self, text: str) -> bool:
        """Check if text looks like a question"""
        return bool(re.search(r'\?|(?:what|how|when|where|why|which|who|do|does|is|are|can|will|should|have)\b', text, re.IGNORECASE))
    
    def _create_question(self, q_id: int, question_text: str, options: List[str]) -> Question:
        """Create a Question object"""
        # Ensure we have at least 2 options, pad with empty if needed
        while len(options) < 2:
            options.append("")
        
        # Limit to 6 options maximum
        options = options[:6]
        
        return Question(
            q_id=q_id,
            question=question_text,
            options=[Option(option_id=i+1, opt_value=opt) for i, opt in enumerate(options)],
            age_group="All",
            gender="Both"
        )

# ===================== TEXT QUESTION PROCESSOR =====================

class TextQuestionProcessor:
    """Main interface for text-based question processing"""
    
    def __init__(self):
        self.parser = UniversalQuestionParser()
    
    def detect_format(self, text: str) -> str:
        """Detect the format of input text"""
        return self.parser.detect_format(text)
    
    def parse_questions(self, text: str) -> List[Question]:
        """Parse questions from text"""
        return self.parser.parse_questions(text)
    
    def parse_direct_input(self, text: str) -> List[Question]:
        """Parse questions directly from user input text"""
        try:
            return self.parser.parse_questions(text)
        except Exception as e:
            logger.error(f"Error in parse_direct_input: {e}")
            return []
    
    def extract_questions_from_response(self, text_response: str) -> List[Question]:
        """Extract questions from processor response (for CSV/file processing)"""
        try:
            return self.parser.parse_questions(text_response)
        except Exception as e:
            logger.error(f"Error in extract_questions_from_response: {e}")
            return []
    
    def get_format_examples(self) -> str:
        """Get examples of supported text formats"""
        return """
**Supported Text Formats:**

**Format 1: Standard Lettered (Multi-line)**
1. What causes fever in children?
A) Viral infection
B) Bacterial infection
C) Immunization
D) All of the above

**Format 2: Numbered Options (Multi-line)**
Question 1: Which organ filters blood?
1) Heart
2) Liver
3) Kidney
4) Lung

**Format 3: Single Line Space-Separated**
2. What is normal body temperature? 96.8°F  97.8°F  98.6°F  99.6°F

**Format 4: Mixed Delimiters**
3. Do you smoke? | Yes regularly | Yes occasionally | No quit recently | Never smoked

**Format 5: Simple Multi-line (No markers)**
4. How often do you exercise?
Daily
Weekly  
Monthly
Never

**Format 6: Comma-Separated**
5. What is your pain level?, Severe, Moderate, Mild, None
"""

# ===================== MAIN FUNCTIONS =====================

def has_questions_and_options(text: str) -> bool:
    """Production-ready function to detect if text contains questions and options"""
    if not text or len(text.strip()) < 5:
        return False
    
    try:
        parser = UniversalQuestionParser()
        detected_format = parser.detect_format(text)
        
        # If we detect any format or can parse questions, return True
        if detected_format != "unknown":
            return True
        
        # Try parsing to see if we get any questions
        questions = parser.parse_questions(text)
        return len(questions) > 0
    except Exception as e:
        logger.error(f"Error in has_questions_and_options: {e}")
        # If there's an error, return True to let the main parser handle it
        return '?' in text and len(text.split()) > 3

# ===================== CLI INTERFACE =====================

def main():
    """CLI interface for testing"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Text-Based Question Processor")
    parser.add_argument("--text", help="Text containing questions and options")
    parser.add_argument("--file", help="Read text from file instead")
    parser.add_argument("--examples", action="store_true", help="Show format examples")
    parser.add_argument("--test", action="store_true", help="Run test cases")
    
    args = parser.parse_args()
    
    processor = TextQuestionProcessor()
    
    if args.examples:
        print("Format Examples:")
        print(processor.get_format_examples())
        return
    
    if args.test:
        test_all_formats()
        return
    
    try:
        # Get text input
        if args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text_input = f.read()
            print(f"Processing text from file: {args.file}")
        elif args.text:
            text_input = args.text
            print("Processing provided text...")
        else:
            print("Please provide --text or --file argument")
            return
        
        # Test format detection
        detected_format = processor.detect_format(text_input)
        has_questions = has_questions_and_options(text_input)
        
        print(f"Detected format: {detected_format}")
        print(f"Has questions and options: {has_questions}")
        
        # Parse questions
        questions = processor.parse_questions(text_input)
        
        if questions:
            print(f"\n=== Found {len(questions)} questions ===")
            for i, q in enumerate(questions, 1):
                print(f"\nQuestion {i}: {q.question}")
                for opt in q.options:
                    if opt.opt_value:
                        print(f"  {opt.option_id}) {opt.opt_value}")
        else:
            print("No questions found")
            
    except Exception as e:
        print(f"Error: {e}")

def test_all_formats():
    """Test function to verify all formats work"""
    parser = UniversalQuestionParser()
    
    test_cases = [
        # Format 1: Standard lettered
        """1. Do you experience shortness of breath?
A) Yes, frequently
B) Yes, occasionally  
C) No
D) Only during exercise""",
        
        # Format 2: Numbered
        """Question 1: How long have you had the cough?
1) Less than 1 week
2) 1-2 weeks
3) More than 2 weeks""",
        
        # Format 3: Single line space-separated
        "2. Are you currently experiencing chest pain? Yes, sharp pain  Yes, dull ache  No pain",
        
        # Format 4: Mixed delimiters
        "3. Have you had a fever recently? | Yes, past 24 hours | Yes, 3-5 days | No",
        
        # Format 5: Simple multi-line
        """4. Does your cough worsen at any time?
Yes, at night
Yes, in the morning  
No specific pattern""",
        
        # The user's exact format
        "1. Do you experience shortness of breath? Yes, frequently Yes, occasionally No 2. How long have you had the cough? Less than 1 week 1–2 weeks More than 2 weeks"
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n=== Testing Format {i} ===")
        print(f"Input: {test_case[:50]}...")
        questions = parser.parse_questions(test_case)
        print(f"Parsed: {len(questions)} questions")
        if questions:
            for j, q in enumerate(questions):
                print(f"  Q{j+1}: {q.question[:40]}...")
                print(f"    Options: {[opt.opt_value for opt in q.options if opt.opt_value]}")

if __name__ == "__main__":
    main()