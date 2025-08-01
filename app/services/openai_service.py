import openai
from typing import Dict, Any, Optional
import json
import base64
from app.core.config import settings

openai.api_key = settings.OPENAI_API_KEY


class OpenAIService:
    @staticmethod
    async def analyze_document(file_content: bytes, file_type: str, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Analyze document content using OpenAI Vision API
        """
        try:
            # Convert file content to base64 for OpenAI API
            base64_content = base64.b64encode(file_content).decode('utf-8')
            
            # Prepare the prompt
            prompt = """
            Analyze the given image/document and provide a structured response:

            1. First, detect the language of the content
            2. Identify if this contains MCQ (Multiple Choice Questions) or written questions
            3. Provide a short title for the page/content
            4. Provide a summary title for your answer
            5. Based on the question type:
               - If MCQ: Extract questions and provide them in the specified JSON format
               - If written: Provide organized explanatory content

            Respond in the detected language and format your response as JSON with this structure:
            {
                "type": "mcq" or "written" or "other",
                "language": "detected language",
                "title": "short title for the content",
                "summary_title": "summary of your response",
                "token": number_of_tokens_used,
                "_result": {
                    // For MCQ type:
                    "questions": [
                        {
                            "question": "question text",
                            "options": {"a": "option1", "b": "option2", "c": "option3", "d": "option4"},
                            "answer": "correct option letter or N/A",
                            "explanation": "brief explanation"
                        }
                    ]
                    // For written type:
                    "content": "organized explanatory text as you would provide in a chat response"
                }
            }
            """
            
            # Make API call to OpenAI
            response = await openai.ChatCompletion.acreate(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{file_type};base64,{base64_content}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            # Parse the response
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            try:
                # Try to parse as JSON
                parsed_response = json.loads(content)
                parsed_response["token"] = tokens_used
                return parsed_response
            except json.JSONDecodeError:
                # If not valid JSON, wrap in a standard format
                return {
                    "type": "other",
                    "language": "unknown",
                    "title": "Document Analysis",
                    "summary_title": "Analysis Result",
                    "token": tokens_used,
                    "_result": {
                        "content": content
                    }
                }
                
        except Exception as e:
            return {
                "type": "error",
                "language": "unknown",
                "title": "Analysis Failed",
                "summary_title": "Error",
                "token": 0,
                "_result": {
                    "error": str(e)
                }
            }

    @staticmethod
    def calculate_points_cost(tokens_used: int) -> int:
        """
        Calculate points cost based on tokens used
        For now, 1 point = 100 tokens (adjust as needed)
        """
        return max(1, tokens_used // 100)