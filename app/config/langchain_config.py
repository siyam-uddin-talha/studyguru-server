"""
LangChain configuration for StudyGuru Pro
"""

import os
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.config.cache_manager import cache_manager
from langchain_core.embeddings import Embeddings
from typing import List
import numpy as np
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import json
import re
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from app.core.config import settings


class CompatibleEmbeddings(Embeddings):
    """Embeddings wrapper that ensures compatibility between different embedding models"""

    def __init__(self, base_embeddings: Embeddings, target_dimension: int = 1536):
        self.base_embeddings = base_embeddings
        self.target_dimension = target_dimension
        self.source_dimension = self._get_source_dimension()

    def _get_source_dimension(self) -> int:
        """Get the source dimension of the base embeddings"""
        if isinstance(self.base_embeddings, GoogleGenerativeAIEmbeddings):
            return 768  # Gemini embedding-001 dimension
        elif isinstance(self.base_embeddings, OpenAIEmbeddings):
            return 1536  # text-embedding-3-small dimension
        else:
            return 1536  # Default assumption

    def _pad_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Pad embeddings to target dimension if needed"""
        if self.source_dimension >= self.target_dimension:
            return embeddings

        padded_embeddings = []
        for embedding in embeddings:
            if len(embedding) < self.target_dimension:
                # Pad with zeros to reach target dimension
                padding = [0.0] * (self.target_dimension - len(embedding))
                padded_embedding = embedding + padding
                padded_embeddings.append(padded_embedding)
            else:
                padded_embeddings.append(embedding)

        return padded_embeddings

    def _truncate_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Truncate embeddings to source dimension if needed"""
        if self.source_dimension >= self.target_dimension:
            return embeddings

        truncated_embeddings = []
        for embedding in embeddings:
            if len(embedding) > self.source_dimension:
                # Truncate to source dimension
                truncated_embedding = embedding[: self.source_dimension]
                truncated_embeddings.append(truncated_embedding)
            else:
                truncated_embeddings.append(embedding)

        return truncated_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with dimension compatibility"""
        embeddings = self.base_embeddings.embed_documents(texts)
        return self._pad_embeddings(embeddings)

    def embed_query(self, text: str) -> List[float]:
        """Embed query with dimension compatibility"""
        embedding = self.base_embeddings.embed_query(text)
        padded_embeddings = self._pad_embeddings([embedding])
        return padded_embeddings[0]


class MarkdownJsonOutputParser(JsonOutputParser):
    """Custom JSON parser that can handle JSON wrapped in markdown code blocks"""

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse JSON from text, handling markdown code blocks and various formats"""
        if not text or not text.strip():
            raise ValueError("Empty or whitespace-only text provided")

        # Clean the text first
        text = text.strip()

        try:
            # First try to parse as regular JSON
            return json.loads(text)
        except json.JSONDecodeError:
            # If that fails, try multiple extraction strategies
            extraction_strategies = [
                # Strategy 1: Look for JSON in markdown code blocks
                lambda t: re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL),
                # Strategy 2: Look for JSON in code blocks without language specifier
                lambda t: re.search(r"```\s*(\{.*?\})\s*```", t, re.DOTALL),
                # Strategy 3: Look for JSON object in the text (most permissive)
                lambda t: re.search(r"\{.*\}", t, re.DOTALL),
                # Strategy 4: Look for JSON array in the text
                lambda t: re.search(r"\[.*\]", t, re.DOTALL),
            ]

            for strategy in extraction_strategies:
                try:
                    json_match = strategy(text)
                    if json_match:
                        json_str = (
                            json_match.group(1)
                            if json_match.groups()
                            else json_match.group(0)
                        )
                        # Clean the extracted JSON string
                        json_str = json_str.strip()
                        return json.loads(json_str)
                except (json.JSONDecodeError, AttributeError):
                    continue

            # If all strategies fail, try to find any valid JSON structure
            try:
                # Look for the first complete JSON object or array
                for i, char in enumerate(text):
                    if char in "{[":
                        # Try to find the matching closing bracket
                        bracket_count = 0
                        start_bracket = char
                        end_bracket = "}" if char == "{" else "]"

                        for j in range(i, len(text)):
                            if text[j] == start_bracket:
                                bracket_count += 1
                            elif text[j] == end_bracket:
                                bracket_count -= 1
                                if bracket_count == 0:
                                    # Found complete JSON structure
                                    json_str = text[i : j + 1]
                                    return json.loads(json_str)
            except (json.JSONDecodeError, IndexError):
                pass

            # If all else fails, raise a descriptive error
            raise ValueError(
                f"Could not parse JSON from text. Text preview: {text[:200]}..."
            )


class StudyGuruPrompts:
    """Centralized prompt templates for StudyGuru"""

    DOCUMENT_ANALYSIS = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
        You are StudyGuru AI analyzing educational content. Analyze the given image/document and provide a structured response:

CONTENT TYPE DETECTION:
- "mcq": If you see ANY of these patterns:
  * Questions with lettered sub-parts (a), b), c), d), etc.)
  * Numbered exercises with multiple parts
  * Multiple choice questions with A, B, C, D options
  * Practice problems with sub-questions
  * Exercises that can be broken into separate solvable problems
  * Exercise sheets with multiple problems to solve
  * Math worksheets with numbered questions (1, 2, 3, etc.)
  * Quiz papers with multiple questions
  * Any document with multiple separate questions that can be answered individually
  
- "written": Only if it's pure explanatory text, essays, or single concept explanations
- "other": For mixed content or unclear format

CRITICAL: If you see a document with multiple numbered questions (like 1, 2, 3, 4, 5, 6, 7, 8, 9), it MUST be classified as "mcq" type, even if the questions don't have traditional A/B/C/D options. Each numbered question should be treated as a separate question to extract.

INSTRUCTIONS:
1. First, detect the language of the content
2. Based on the question type:
   - If MCQ: Extract each question/exercise part as a separate question
     * Look for numbered questions (1, 2, 3, 4, 5, 6, 7, 8, 9, etc.) and extract each one
     * If the document contains actual multiple choice options (like: A, B, C, D or a, b, c, d or 1, 2, 3, 4), include them in the "options" field
     * If the document does NOT contain multiple choice options, omit the "options" field entirely
     * For the "answer" field: provide the correct option letter if options exist, or provide the actual solution/answer if no options are given
     * IMPORTANT: Extract ALL numbered questions from the document, not just a summary
   - If written: Provide organized explanatory content

RESPONSE FORMAT:
Respond with valid JSON only. No additional text or formatting.

For MCQ content:
{{
    "type": "mcq",
    "language": "detected language",
    "_result": {{
        "questions": [
            {{
                "question": "question text (e.g., 'Domain of 1/x is …………….')",
                "options": {{
                    "a": "option1",
                    "b": "option2", 
                    "c": "option3",
                    "d": "option4"
                }},
                "answer": "correct option letter (like: 'c')",
                "explanation": "step-by-step solution or brief explanation"
            }}
        ]
    }}
}}

For written content:
{{
    "type": "written",
    "language": "detected language", 
    
    "_result": {{
        "content": "organized explanatory text as you would provide in a chat response"
    }}
}}

CRITICAL REQUIREMENTS:
- If the document has multiple choice options, ALWAYS include the "options" field with a, b, c, d keys
- If the document does NOT have multiple choice options, omit the "options" field completely
- The "answer" field should contain the correct option letter (a, b, c, d) if options exist
- The "explanation" field should provide clear, educational explanations
- Extract ALL questions from the document - don't summarize or skip any
- Use proper mathematical notation and clear language
- CRITICAL: Respond with valid JSON only. No markdown, no explanations outside JSON.
        """,
            ),
            (
                "human",
                [
                    {"type": "text", "text": "Please analyze this document/image:"},
                    {"type": "image_url", "image_url": {"url": "{file_url}"}},
                ],
            ),
        ]
    )

    GUARDRAIL_CHECK = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Educational content guardrail. Check if content violates rules.

REJECT: Non-educational content (selfies, social media, inappropriate material, code generation requests, personal photos)
ACCEPT: Educational content (textbooks, worksheets, math problems, study notes, academic papers, quizzes, question papers, mcq papers, exam papers, homework, assignments, educational diagrams, mathematical equations, scientific content)

CRITICAL RULES:
1. If you see ANY educational content (math problems, questions, equations, academic text), ACCEPT it
2. MCQ papers, question papers, and exam papers are ALWAYS educational content
3. Mathematical equations, formulas, and problems are educational content
4. Textbooks, worksheets, and study materials are educational content
5. Even if there are faces in the image, if educational content is present, ACCEPT it
6. Empty or minimal content should be ACCEPTED (not rejected)
7. Only reject if content is clearly non-educational (selfies, social media, inappropriate material)

JSON response only:
{{
    "is_violation": false,
    "violation_type": null,
    "reasoning": "Educational content detected: [brief description]"
}}

For violations:
{{
    "is_violation": true,
    "violation_type": "non_educational_content", 
    "reasoning": "Brief explanation of why content is non-educational"
}}
                """,
            ),
            ("human", "{content}"),
        ]
    )

    CONVERSATION_WITH_CONTEXT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are StudyGuru AI, an educational assistant with access to user's learning history.

Topic: {interaction_title}
Context: {interaction_summary}

INSTRUCTIONS:
- Use provided context to personalize responses
- Reference previous discussions and build upon existing knowledge
- For specific question numbers (e.g., "mcq 3"), find and answer that exact question from context
- Maintain consistency with user's learning style

RESPONSE FORMAT:
- Plain text only (no JSON)
- Use ### for section headers
- For MCQs: Number questions (1., 2., etc.), list options (A., B., C., D.) if present
- Answer format: "Answer: [solution]" and "Explanation: [text]"
- Use bullet points (•) for lists

Be educational, helpful, and reference context when relevant.
                """,
            ),
            ("human", "{content}"),
        ]
    )

    CONVERSATION_WITHOUT_CONTEXT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are StudyGuru AI, an advanced educational assistant. Format your responses for maximum clarity and educational impact.

FORMATTING GUIDELINES:
1. Use clear section headers with ### for main topics
2. For MCQ content:
   - Number each question clearly (1., 2., etc.)
   - If options exist: List options in format: A. [option], B. [option], etc.
   - If no options exist: Provide the solution directly without listing options
   - Provide answers as "Answer: [letter or solution]" (without asterisks or bold)
   - Include explanations as "Explanation: [detailed explanation]" (without asterisks or bold)
3. Use bullet points with • for lists and key concepts
4. Use plain text formatting - avoid LaTeX, special symbols, or complex markdown
5. Structure information with clear paragraph breaks
6. Write mathematical expressions in plain text (e.g., "x squared" instead of x^2)

EDUCATIONAL APPROACH:
- Provide step-by-step explanations
- Use relevant examples
- Connect theory to practice
- Encourage deeper understanding
- Maintain clarity and precision

Be encouraging, professional, and focused on helping students learn effectively.
                """,
            ),
            ("human", "{content}"),
        ]
    )

    TITLE_GENERATION = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a title generator. Generate a short, descriptive title (max 50 characters) and a summary title (max 100 characters) for the educational content.

Rules:
1. Title should be concise and capture the main topic
2. Summary title should describe what help is being provided
3. Use simple, clear language
4. Avoid special characters or complex formatting
5. Focus on the educational subject matter

Respond in valid JSON format:
{{
    "title": "short descriptive title",
    "summary_title": "what help is being provided"
}}
                """,
            ),
            (
                "human",
                "User message: {message}\nFirst few lines of response: {response_preview}",
            ),
        ]
    )

    CONVERSATION_SUMMARIZATION = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert at extracting key facts and creating semantic summaries from educational conversations.

Your task is to analyze a conversation between a user and AI, then extract:
1. Key facts and important information discussed
2. Main concepts and topics covered
3. Problems solved or questions answered
4. Important formulas, equations, or rules mentioned
5. Context that would be useful for future conversations
6. Specific question numbers or problems referenced
7. Learning progress indicators
8. Areas where the user might need more help

Create a concise but comprehensive summary that captures the essence of what was discussed.
Focus on educational content, not conversational fluff.

IMPORTANT: Pay special attention to:
- Numbered questions, problems, or equations mentioned
- Specific concepts the user is learning
- Areas where the user showed confusion or needed clarification
- Solutions provided and explanations given
- Any follow-up questions that might arise

CRITICAL REQUIREMENTS:
1. You MUST respond with valid JSON only
2. Do NOT wrap your response in markdown code blocks (```json)
3. Do NOT include any text before or after the JSON
4. Use only plain text - avoid LaTeX notation or special characters
5. Ensure all strings are properly escaped for JSON
6. If the conversation is empty or unclear, still return valid JSON with default values

REQUIRED JSON FORMAT:
{{
    "key_facts": [
        "fact 1: specific information learned with context",
        "fact 2: important concept discussed with details",
        "fact 3: problem solved or explained with solution"
    ],
    "main_topics": ["topic1", "topic2", "topic3"],
    "semantic_summary": "A detailed 3-4 sentence summary capturing the essence of the conversation, what was discussed, and its educational value. Must be at least 50 characters long.",
    "important_terms": ["term1", "term2", "term3"],
    "context_for_future": "What context would be most useful for understanding follow-up questions in this conversation",
    "question_numbers": [1, 2, 3],
    "learning_progress": "What the user has learned or is learning",
    "potential_follow_ups": ["follow-up question 1", "follow-up question 2"],
    "difficulty_level": "beginner|intermediate|advanced",
    "subject_area": "math|science|language|other"
}}

REMEMBER: Your response must be valid JSON that can be parsed directly. No additional text or formatting.
                """,
            ),
            (
                "human",
                "User message: {user_message}\n\nAI response: {ai_response}",
            ),
        ]
    )

    INTERACTION_SUMMARY_UPDATE = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert at maintaining a running semantic summary of an educational conversation.

You will be given:
1. The current accumulated summary of previous conversations
2. A new conversation exchange that just happened

Your task is to create an UPDATED summary that:
- Incorporates new key facts and topics from the latest exchange
- Maintains important context from previous conversations
- Removes redundant or less important information to stay concise
- Prioritizes the most recent and most relevant information
- Keeps the summary under 500 words
- Tracks learning progress and difficulty progression
- Identifies patterns in user questions and learning style

The summary should be optimized for helping the AI understand context in future conversations.

IMPORTANT: Pay special attention to:
- Question numbers and problem references
- Learning progression and difficulty changes
- Areas where the user consistently needs help
- Concepts that are building upon each other
- User's preferred learning style and pace

CRITICAL: Respond ONLY with valid JSON. Do NOT wrap your response in markdown code blocks (```json). Return pure JSON only.
{{
    "updated_summary": "The comprehensive running summary incorporating all important information",
    "key_topics": ["all important topics covered so far"],
    "recent_focus": "What the user has been focusing on most recently (last 2-3 exchanges)",
    "accumulated_facts": ["critical facts that should be remembered for future conversations"],
    "question_numbers": [1, 2, 3, 4, 5],
    "learning_progression": "How the user's understanding has evolved",
    "difficulty_trend": "beginner|intermediate|advanced",
    "learning_patterns": ["pattern1", "pattern2"],
    "struggling_areas": ["area1", "area2"],
    "mastered_concepts": ["concept1", "concept2"]
}}
                """,
            ),
            (
                "human",
                "Current accumulated summary: {current_summary}\n\nNew user message: {new_user_message}\n\nNew AI response: {new_ai_response}",
            ),
        ]
    )

    MCQ_GENERATION = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are StudyGuru AI, an expert at creating high-quality multiple choice questions (MCQs) for educational content.

TASK: Generate well-structured MCQs based on the given topic.

REQUIREMENTS:
1. Create questions that test understanding, not memorization
2. Make questions clear, concise, and unambiguous
3. Provide exactly 4 options (a, b, c, d) for each question
4. Ensure only one correct answer per question
5. Make distractors plausible but clearly incorrect
6. Provide clear, educational explanations
7. Use proper mathematical notation and clear language
8. Vary difficulty levels appropriately
9. Cover different aspects of the topic

OUTPUT FORMAT:
Respond with valid JSON only. No additional text or formatting.

{{
    "type": "mcq",
    "language": "English",
    "title": "Topic Name – Multiple Choice Questions",
    "summary_title": "Solved MCQs with answers and explanations",
    "_result": {{
        "questions": [
            {{
                "question": "Question text here?",
                "options": {{
                    "a": "First option",
                    "b": "Second option",
                    "c": "Third option",
                    "d": "Fourth option"
                }},
                "answer": "correct option letter (a, b, c, or d)",
                "explanation": "Clear explanation of why this answer is correct and why others are wrong"
            }}
        ]
    }}
}}

GENERATION RULES:
- Generate 5-10 high-quality questions per request
- Make questions progressively more challenging
- Ensure explanations are educational and help learning
- Use consistent formatting and clear language
- Test different concepts within the topic
- CRITICAL: Respond with valid JSON only. No markdown, no explanations outside JSON.
                """,
            ),
            (
                "human",
                "Generate multiple choice questions for: {topic_or_content}",
            ),
        ]
    )


class StudyGuruModels:
    """Model configurations for StudyGuru - supports both GPT and Gemini models"""

    # Fallback models in case GPT-5 is not available
    USE_FALLBACK_MODELS = True  # Set to False when GPT-5 is available

    @staticmethod
    def _is_gemini_model() -> bool:
        """Check if Gemini model is configured"""
        return settings.LLM_MODEL.lower() == "gemini"

    @staticmethod
    def get_chat_model(
        temperature: float = 0.2,
        max_tokens: int = 5000,
        reasoning_effort: str = "low",  # Reduced from "medium" for faster responses
        verbosity: str = "low",
    ):
        """Get configured chat model - supports both GPT and Gemini"""
        if StudyGuruModels._is_gemini_model():
            # Gemini 2.5 Pro configuration (equivalent to GPT-4o)
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",  # Using stable Gemini model
                temperature=temperature,
                google_api_key=settings.GOOGLE_API_KEY,
                max_output_tokens=max_tokens,
                request_timeout=30,  # Reduced timeout for faster responses
                cache=cache_manager.get_response_cache(),  # Enable response caching
            )
        else:
            # GPT models
            if StudyGuruModels.USE_FALLBACK_MODELS:
                # Fallback to GPT-4o for compatibility
                return ChatOpenAI(
                    model="gpt-4o",  # Fallback model
                    temperature=temperature,
                    openai_api_key=settings.OPENAI_API_KEY,
                    max_tokens=max_tokens,
                    request_timeout=30,
                    cache=cache_manager.get_response_cache(),  # Enable response caching
                )
            else:
                # GPT-5 configuration - optimized for speed
                return ChatOpenAI(
                    model="gpt-5",  # Latest and most advanced model
                    temperature=temperature,
                    openai_api_key=settings.OPENAI_API_KEY,
                    max_tokens=max_tokens,
                    request_timeout=60,  # Reduced from 120 to 60 seconds for faster responses
                    verbosity="low",  # Always use low verbosity for speed
                    cache=cache_manager.get_response_cache(),  # Enable response caching
                )

    @staticmethod
    def get_vision_model(
        temperature: float = 0.3, max_tokens: int = 5000, verbosity: str = "low"
    ):
        """Get configured vision model - supports both GPT and Gemini"""
        if StudyGuruModels._is_gemini_model():
            # Gemini 2.5 Pro with vision capabilities
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",  # Gemini with vision support
                temperature=temperature,
                google_api_key=settings.GOOGLE_API_KEY,
                max_output_tokens=max_tokens,
                request_timeout=30,
                cache=cache_manager.get_response_cache(),  # Enable response caching
            )
        else:
            # GPT models
            if StudyGuruModels.USE_FALLBACK_MODELS:
                # Fallback to GPT-4o for compatibility
                return ChatOpenAI(
                    model="gpt-4o",  # Fallback model
                    temperature=temperature,
                    openai_api_key=settings.OPENAI_API_KEY,
                    max_tokens=max_tokens,
                    request_timeout=30,
                    cache=cache_manager.get_response_cache(),  # Enable response caching
                )
            else:
                # GPT-5 configuration - optimized for speed
                return ChatOpenAI(
                    model="gpt-5",  # GPT-5 with enhanced vision capabilities
                    temperature=temperature,
                    openai_api_key=settings.OPENAI_API_KEY,
                    max_tokens=max_tokens,
                    request_timeout=30,  # Reduced timeout for faster processing
                    verbosity="low",  # Low verbosity for faster responses
                    cache=cache_manager.get_response_cache(),  # Enable response caching
                )

    @staticmethod
    def get_guardrail_model(
        temperature: float = 0.1, max_tokens: int = 500, verbosity: str = "low"
    ):
        """Get configured guardrail model - supports both GPT and Gemini"""
        if StudyGuruModels._is_gemini_model():
            # Gemini 2.5 Flash for cost efficiency (equivalent to GPT-4o-mini)
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",  # Fast and cost-effective model
                temperature=temperature,
                google_api_key=settings.GOOGLE_API_KEY,
                max_output_tokens=max_tokens,
                request_timeout=15,
                cache=cache_manager.get_response_cache(),  # Enable response caching
            )
        else:
            # GPT models
            if StudyGuruModels.USE_FALLBACK_MODELS:
                # Fallback to GPT-4o-mini for compatibility
                return ChatOpenAI(
                    model="gpt-4o-mini",  # Fallback model
                    temperature=temperature,
                    openai_api_key=settings.OPENAI_API_KEY,
                    max_tokens=max_tokens,
                    request_timeout=15,
                    cache=cache_manager.get_response_cache(),  # Enable response caching
                )
            else:
                # GPT-5 Mini configuration
                return ChatOpenAI(
                    model="gpt-5-mini",  # GPT-5 Mini: 83% more cost-effective than GPT-5
                    temperature=temperature,
                    openai_api_key=settings.OPENAI_API_KEY,
                    max_tokens=max_tokens,
                    request_timeout=15,  # Fast timeout for guardrails
                    verbosity=verbosity,  # Control response length and detail
                    cache=cache_manager.get_response_cache(),  # Enable response caching
                )

    @staticmethod
    def get_complex_reasoning_model(
        temperature: float = 0.1, max_tokens: int = 5000, verbosity: str = "medium"
    ):
        """Get configured model for complex reasoning tasks - supports both GPT and Gemini"""
        if StudyGuruModels._is_gemini_model():
            # Gemini 2.5 Pro for complex reasoning
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",  # Latest Gemini for complex tasks
                temperature=temperature,
                google_api_key=settings.GOOGLE_API_KEY,
                max_output_tokens=max_tokens,
                request_timeout=60,
                cache=cache_manager.get_response_cache(),  # Enable response caching
            )
        else:
            # GPT models
            if StudyGuruModels.USE_FALLBACK_MODELS:
                # Fallback to GPT-4o for compatibility
                return ChatOpenAI(
                    model="gpt-4o",  # Fallback model
                    temperature=temperature,
                    openai_api_key=settings.OPENAI_API_KEY,
                    max_tokens=max_tokens,
                    request_timeout=60,
                    cache=cache_manager.get_response_cache(),  # Enable response caching
                )
            else:
                # GPT-5 configuration
                return ChatOpenAI(
                    model="gpt-5",  # GPT-5 for complex reasoning
                    temperature=temperature,
                    openai_api_key=settings.OPENAI_API_KEY,
                    max_tokens=max_tokens,
                    request_timeout=150,  # Increased timeout for complex reasoning with high effort
                    verbosity=verbosity,  # Medium verbosity for complex reasoning tasks
                    cache=cache_manager.get_response_cache(),  # Enable response caching
                )

    @staticmethod
    def get_embeddings_model():
        """Get configured embeddings model - supports both GPT and Gemini with compatibility"""
        if StudyGuruModels._is_gemini_model():
            # Gemini embeddings model (fast and cost-efficient)
            base_embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",  # Gemini embedding model
                google_api_key=settings.GOOGLE_API_KEY,
            )
        else:
            # OpenAI embeddings
            base_embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small", openai_api_key=settings.OPENAI_API_KEY
            )

        # Wrap with compatibility layer to ensure 1536 dimensions
        return CompatibleEmbeddings(base_embeddings, target_dimension=1536)

    @staticmethod
    def get_title_model(
        temperature: float = 0.3, max_tokens: int = 100, verbosity: str = "low"
    ):
        """Get configured title generation model - supports both GPT and Gemini"""
        if StudyGuruModels._is_gemini_model():
            # Gemini 2.5 Flash for cost efficiency
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",  # Fast and cost-effective model
                temperature=temperature,
                google_api_key=settings.GOOGLE_API_KEY,
                max_output_tokens=max_tokens,
                request_timeout=10,  # Fast timeout for quick response
                cache=cache_manager.get_response_cache(),  # Enable response caching
            )
        else:
            # GPT models
            if StudyGuruModels.USE_FALLBACK_MODELS:
                # Fallback to GPT-4o-mini for compatibility
                return ChatOpenAI(
                    model="gpt-4o-mini",  # Fallback model
                    temperature=temperature,
                    openai_api_key=settings.OPENAI_API_KEY,
                    max_tokens=max_tokens,  # Very low token limit for cost efficiency
                    request_timeout=10,  # Fast timeout for quick response
                    cache=cache_manager.get_response_cache(),  # Enable response caching
                )
            else:
                # GPT-5 Mini configuration
                return ChatOpenAI(
                    model="gpt-5-mini",  # GPT-5 Mini: Most cost-effective model
                    temperature=temperature,
                    openai_api_key=settings.OPENAI_API_KEY,
                    max_tokens=max_tokens,  # Very low token limit for cost efficiency
                    request_timeout=10,  # Fast timeout for quick response
                    verbosity=verbosity,  # Control response length and detail
                    cache=cache_manager.get_response_cache(),  # Enable response caching
                )

    @staticmethod
    def get_model_with_context_cache(
        model_type: str = "chat",
        temperature: float = 0.2,
        max_tokens: int = 5000,
        cached_content: Optional[Any] = None,
    ):
        """
        Get model with context caching for large documents

        Args:
            model_type: Type of model ("chat", "vision", "guardrail", "reasoning", "title")
            temperature: Model temperature
            max_tokens: Maximum output tokens
            cached_content: Pre-cached content for context caching

        Returns:
            Model instance with context caching enabled
        """
        if StudyGuruModels._is_gemini_model() and cached_content:
            # Use context caching for Gemini models
            if model_type == "chat":
                return ChatGoogleGenerativeAI(
                    model="gemini-2.5-pro",
                    temperature=temperature,
                    google_api_key=settings.GOOGLE_API_KEY,
                    max_output_tokens=max_tokens,
                    cache=cache_manager.get_response_cache(),
                    cache_context=cached_content,  # Enable context caching
                )
            elif model_type == "vision":
                return ChatGoogleGenerativeAI(
                    model="gemini-2.5-pro",
                    temperature=temperature,
                    google_api_key=settings.GOOGLE_API_KEY,
                    max_output_tokens=max_tokens,
                    request_timeout=30,
                    cache=cache_manager.get_response_cache(),
                    cache_context=cached_content,  # Enable context caching
                )
            elif model_type == "guardrail":
                return ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=temperature,
                    google_api_key=settings.GOOGLE_API_KEY,
                    max_output_tokens=max_tokens,
                    request_timeout=15,
                    cache=cache_manager.get_response_cache(),
                    cache_context=cached_content,  # Enable context caching
                )
            elif model_type == "reasoning":
                return ChatGoogleGenerativeAI(
                    model="gemini-2.5-pro",
                    temperature=temperature,
                    google_api_key=settings.GOOGLE_API_KEY,
                    max_output_tokens=max_tokens,
                    request_timeout=60,
                    cache=cache_manager.get_response_cache(),
                    cache_context=cached_content,  # Enable context caching
                )
            elif model_type == "title":
                return ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=temperature,
                    google_api_key=settings.GOOGLE_API_KEY,
                    max_output_tokens=max_tokens,
                    request_timeout=10,
                    cache=cache_manager.get_response_cache(),
                    cache_context=cached_content,  # Enable context caching
                )

        # Fallback to regular model without context caching
        if model_type == "chat":
            return StudyGuruModels.get_chat_model(temperature, max_tokens)
        elif model_type == "vision":
            return StudyGuruModels.get_vision_model(temperature, max_tokens)
        elif model_type == "guardrail":
            return StudyGuruModels.get_guardrail_model(temperature, max_tokens)
        elif model_type == "reasoning":
            return StudyGuruModels.get_complex_reasoning_model(temperature, max_tokens)
        elif model_type == "title":
            return StudyGuruModels.get_title_model(temperature, max_tokens)
        else:
            return StudyGuruModels.get_chat_model(temperature, max_tokens)


class StudyGuruVectorStore:
    """Vector store configuration for StudyGuru"""

    @staticmethod
    def get_milvus_config() -> Dict[str, Any]:
        """Get Milvus connection configuration"""
        return {
            "uri": settings.ZILLIZ_URI,
            "token": settings.ZILLIZ_TOKEN,
            "secure": True,
        }

    @staticmethod
    def get_collection_config() -> Dict[str, Any]:
        """Get collection configuration - uses single collection with common dimension"""
        # Use single collection with 1536 dimensions (largest) for compatibility
        # Gemini embeddings (768D) will be padded to 1536D
        dimension = 1536  # Common dimension for both models
        collection_name = settings.ZILLIZ_COLLECTION  # Single collection

        return {
            "collection_name": collection_name,
            "dimension": dimension,
            "index_params": {
                "index_type": "IVF_FLAT",
                "metric_type": settings.ZILLIZ_INDEX_METRIC,
                "params": {"nlist": 1024},
            },
        }


class StudyGuruChains:
    """
    Pre-configured chains for StudyGuru operations

    Parser Strategy:
    - MarkdownJsonOutputParser: Used for all chains that might receive mixed content
      (AI responses with explanations + JSON, markdown code blocks, etc.)
    - JsonOutputParser: Reserved for simple, pure JSON responses (currently unused)

    The MarkdownJsonOutputParser is more robust and can handle:
    - Pure JSON responses
    - JSON wrapped in markdown code blocks
    - Mixed content with JSON embedded in text
    - Various formatting issues
    """

    @staticmethod
    def get_document_analysis_chain():
        """Get document analysis chain using GPT-5 with high reasoning effort for better analysis"""
        model = StudyGuruModels.get_vision_model()
        parser = MarkdownJsonOutputParser()  # Use robust parser for document analysis
        return StudyGuruPrompts.DOCUMENT_ANALYSIS | model | parser

    @staticmethod
    def get_guardrail_chain():
        """Get guardrail check chain"""
        model = StudyGuruModels.get_guardrail_model(temperature=0.2, max_tokens=400)
        parser = MarkdownJsonOutputParser()  # Use robust parser for guardrails
        return StudyGuruPrompts.GUARDRAIL_CHECK | model | parser

    @staticmethod
    def get_conversation_chain(has_context: bool = False):
        """Get conversation chain"""
        model = StudyGuruModels.get_chat_model()
        parser = StrOutputParser()

        if has_context:
            return StudyGuruPrompts.CONVERSATION_WITH_CONTEXT | model | parser
        else:
            return StudyGuruPrompts.CONVERSATION_WITHOUT_CONTEXT | model | parser

    @staticmethod
    def get_title_generation_chain():
        """Get title generation chain (cost-optimized) with robust error handling"""
        # Use GPT-4o-mini for better JSON stability instead of GPT-5-mini
        # GPT-5 models have issues with response_format parameter
        model = ChatOpenAI(
            model="gpt-4o-mini",  # More reliable for JSON output
            temperature=0.3,
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=300,  # Increased to ensure complete JSON response
            request_timeout=20,  # Increased timeout
            # model_kwargs={"response_format": {"type": "json_object"}},
        )
        parser = MarkdownJsonOutputParser()  # Use robust parser for title generation
        return StudyGuruPrompts.TITLE_GENERATION | model | parser

    @staticmethod
    def get_conversation_summarization_chain():
        """Get conversation summarization chain with increased token limits"""
        # Use GPT-4o-mini for better stability and disable reasoning to reserve tokens for output
        model = ChatOpenAI(
            model="gpt-4o-mini",  # More reliable for JSON output
            temperature=0.2,
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=2000,  # Aggressively increased to handle reasoning + output
            request_timeout=45,  # Increased timeout for longer processing
            # model_kwargs={"response_format": {"type": "json_object"}},
        )
        parser = MarkdownJsonOutputParser()
        return StudyGuruPrompts.CONVERSATION_SUMMARIZATION | model | parser

    @staticmethod
    def get_interaction_summary_update_chain():
        """Get interaction summary update chain with increased token limits"""
        # Use GPT-4o-mini for better stability and increased token limits
        model = ChatOpenAI(
            model="gpt-4o-mini",  # More reliable for JSON output
            temperature=0.2,
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=1500,  # Aggressively increased to handle longer updates
            request_timeout=45,  # Increased timeout
            # model_kwargs={"response_format": {"type": "json_object"}},
        )
        parser = MarkdownJsonOutputParser()
        return StudyGuruPrompts.INTERACTION_SUMMARY_UPDATE | model | parser

    @staticmethod
    def get_mcq_generation_chain():
        """Get MCQ generation chain using complex reasoning model for better question quality"""
        model = StudyGuruModels.get_complex_reasoning_model(
            temperature=0.3, max_tokens=1200
        )
        parser = MarkdownJsonOutputParser()  # Use robust parser for MCQ generation
        return StudyGuruPrompts.MCQ_GENERATION | model | parser


class StudyGuruConfig:
    """Main configuration class for StudyGuru LangChain setup"""

    # Model configurations
    MODELS = StudyGuruModels

    # Prompt templates
    PROMPTS = StudyGuruPrompts

    # Vector store configurations
    VECTOR_STORE = StudyGuruVectorStore

    # Pre-configured chains
    CHAINS = StudyGuruChains

    # Token system configuration
    BASE_TOKENS = 5000  # Base tokens for text-only prompts
    TOKENS_PER_FILE = 5000  # Additional tokens per file
    MAX_TOKENS_LIMIT = 20000  # Maximum token limit to prevent excessive usage

    # Default settings
    DEFAULT_MAX_TOKENS = 5000
    DEFAULT_TEMPERATURE = 0.2
    DEFAULT_TOP_K = 5

    # Points calculation
    POINTS_PER_TOKEN = 100

    @staticmethod
    def calculate_dynamic_tokens(file_count: int = 0) -> int:
        """
        Calculate dynamic token limit based on file count.

        Args:
            file_count: Number of files being processed

        Returns:
            int: Calculated token limit (BASE_TOKENS + TOKENS_PER_FILE * file_count)
        """
        calculated_tokens = StudyGuruConfig.BASE_TOKENS + (
            StudyGuruConfig.TOKENS_PER_FILE * file_count
        )
        return min(calculated_tokens, StudyGuruConfig.MAX_TOKENS_LIMIT)

    @staticmethod
    def calculate_points_cost(tokens_used: int) -> int:
        """Calculate points cost based on tokens used"""
        return max(1, tokens_used // StudyGuruConfig.POINTS_PER_TOKEN)

    @staticmethod
    def get_retrieval_config(user_id: str, top_k: int = None) -> Dict[str, Any]:
        """Get retrieval configuration for a specific user"""
        return {
            "k": top_k or StudyGuruConfig.DEFAULT_TOP_K,
            "filter": f"user_id == '{user_id}'",
        }
