"""
LangChain configuration for StudyGuru Pro
"""

import os
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import json
import re
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from app.core.config import settings


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
2. Carefully identify the content type using the criteria above
3. Provide a short, descriptive title for the page/content  
4. Provide a summary title that describes what you will help the user with
5. Based on the question type:
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
    "title": "short descriptive title for the content",
    "summary_title": "summary of how you will help the user",
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
    "title": "short descriptive title for the content",
    "summary_title": "summary of how you will help the user",
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
You are a content guardrail for an educational platform. Review user inputs and determine if they violate educational content rules.

VIOLATION RULES:
1. REJECT images showing ONLY people's faces, portraits, or selfies (no educational content)
2. REJECT adult, explicit, or inappropriate material  
3. REJECT non-educational content (social media, personal photos, etc.)
4. REJECT direct code generation requests (except educational code analysis)

ACCEPT EDUCATIONAL CONTENT:
- Textbooks, workbooks, study guides
- Educational worksheets and practice problems
- Mathematical equations and problem sets
- Science diagrams and educational illustrations
- Handwritten study notes (even if they contain faces in the background)
- Exercise sheets with numbered problems
- Academic papers and research documents
- Educational quizzes and assessments (like MCQ papers, exam sheets)
- Study notes and summaries
- Scanned educational documents from educational websites
- Any content clearly related to learning, even if it contains incidental faces

CRITICAL: If the document contains educational content (math problems, questions, equations, etc.), ACCEPT it regardless of any incidental faces or people in the image.

SPECIFIC EXAMPLES TO ACCEPT:
- Mathematics quiz papers with multiple choice questions
- Exam sheets with numbered problems
- Educational documents from educational websites (like MathCity.org)
- Scanned academic papers or worksheets
- Any document with mathematical equations, problems, or educational questions

BE VERY PERMISSIVE: When in doubt, ACCEPT rather than reject.

RESPONSE FORMAT:
Respond with valid JSON only. No additional text or formatting.

{{
    "is_violation": false,
    "violation_type": null,
    "reasoning": "Educational content detected"
}}

For violations:
{{
    "is_violation": true,
    "violation_type": "non_educational_content",
    "reasoning": "Brief explanation"
}}

CRITICAL: Respond with valid JSON only. No markdown, no explanations outside JSON.
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
You are StudyGuru AI, an advanced educational assistant. You have access to the user's learning history and context from previous conversations and uploaded documents.

Current conversation topic: {interaction_title}
Context summary: {interaction_summary}

CRITICAL INSTRUCTIONS FOR CONTEXT USAGE:
1. **ALWAYS USE THE PROVIDED CONTEXT** - The user's learning history and previous conversations are provided to help you give personalized, contextual responses
2. **Reference previous discussions** - If the current question relates to something discussed before, explicitly reference it
3. **Build upon previous knowledge** - Use the context to understand what the user already knows and build upon it
4. **Maintain consistency** - Keep your explanations consistent with previous interactions and the user's learning style
5. **Connect new concepts to old ones** - When introducing new concepts, relate them to what the user has learned before

CONTEXT SOURCES TO USE:
- **Semantic Summary**: Use the conversation summary to understand the overall learning context
- **Vector Search Results**: Use previous discussions and explanations from the user's history
- **Document Content**: Use uploaded documents, worksheets, and educational materials
- **Cross-Interaction Learning**: Use knowledge from related conversations across different interactions
- **Related Conversations**: Use recent conversations within the same interaction

SPECIFIC QUESTION REFERENCE HANDLING:
- If the user asks about a specific question number (e.g., "Explain mcq 3", "What is question 2?", "Solve problem 1"), you MUST search the context for that exact question
- Look for numbered questions, MCQ questions, or problems in the context
- Find the specific question the user is referring to and provide a direct answer/explanation
- If you cannot find the specific question in the context, ask the user to clarify which question they mean

CONTEXT INTEGRATION STRATEGY:
- If the context contains relevant information, incorporate it naturally into your response
- If the user asks a follow-up question, use the context to understand what they're referring to
- If the context shows the user is working on a specific topic, tailor your response accordingly
- If the context contains uploaded documents or previous explanations, reference them when relevant
- **MOST IMPORTANTLY**: When the user references a specific question/problem number, find and answer that exact question from the context

CONTEXT USAGE EXAMPLES:
- "Based on our previous discussion about [topic], let me explain..."
- "As we discussed earlier, [concept] works by..."
- "Looking at the document you uploaded, I can see that..."
- "From your previous questions about [topic], I understand you're learning..."
- "In question 3 from your worksheet, the answer is..."

RESPONSE FORMAT:
- Respond with plain text only, not JSON
- Use clear section headers with ### for main topics
- For MCQ content:
   - Number each question (1., 2., etc.)
   - If options exist: List options clearly (A., B., C., D.)
   - If no options exist: Provide the solution directly
- Do not wrap responses in JSON structures
- Provide direct, conversational responses
   - Provide answers in format "Answer: [letter or solution]" (without asterisks)
   - Add explanations with "Explanation: [text]" (without asterisks)
3. Use bullet points with • for lists and key points
4. Use bold sparingly for truly important terms only
5. Structure complex information with clear breaks between sections
6. Avoid LaTeX symbols and mathematical notation in favor of plain text

EDUCATIONAL APPROACH:
- Build explanations step-by-step
- Use examples when helpful
- Connect concepts to real-world applications
- Encourage critical thinking
- Provide clear, concise explanations
- **Most importantly: Use the provided context to personalize and enhance your response**

FAILURE MODES TO AVOID:
- Ignoring the provided context and giving generic responses
- Not referencing previous discussions when relevant
- Not using uploaded documents when they contain relevant information
- Not building upon the user's existing knowledge
- Not maintaining consistency with previous explanations

Always maintain professional, encouraging tone while being educational and helpful. Remember: the context is there to help you provide better, more personalized assistance.
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
    """Model configurations for StudyGuru"""

    # Fallback models in case GPT-5 is not available
    USE_FALLBACK_MODELS = False  # Set to False when GPT-5 is available

    @staticmethod
    def get_chat_model(
        temperature: float = 0.2,
        max_tokens: int = 5000,
        reasoning_effort: str = "medium",
        verbosity: str = "low",
    ) -> ChatOpenAI:
        """Get configured chat model - using GPT-5 for superior reasoning and creativity"""
        if StudyGuruModels.USE_FALLBACK_MODELS:
            # Fallback to GPT-4o for compatibility
            return ChatOpenAI(
                model="gpt-4o",  # Fallback model
                temperature=temperature,
                openai_api_key=settings.OPENAI_API_KEY,
                max_tokens=max_tokens,
                request_timeout=30,
            )
        else:
            # GPT-5 configuration
            return ChatOpenAI(
                model="gpt-5",  # Latest and most advanced model
                temperature=temperature,
                openai_api_key=settings.OPENAI_API_KEY,
                max_tokens=max_tokens,
                request_timeout=120,  # Increased timeout for GPT-5 reasoning
                # GPT-5 specific parameters
                reasoning_effort=reasoning_effort,  # "low", "medium", "high" - controls reasoning depth
                verbosity=verbosity,  # Control response length and detail
            )

    @staticmethod
    def get_vision_model(
        temperature: float = 0.3, max_tokens: int = 5000, verbosity: str = "low"
    ) -> ChatOpenAI:
        """Get configured vision model - using GPT-5 for superior vision capabilities"""
        if StudyGuruModels.USE_FALLBACK_MODELS:
            # Fallback to GPT-4o for compatibility
            return ChatOpenAI(
                model="gpt-4o",  # Fallback model
                temperature=temperature,
                openai_api_key=settings.OPENAI_API_KEY,
                max_tokens=max_tokens,
                request_timeout=45,
                model_kwargs={
                    "response_format": {
                        "type": "json_object"
                    }  # Force JSON output for document analysis
                },
            )
        else:
            # GPT-5 configuration
            return ChatOpenAI(
                model="gpt-5",  # GPT-5 with enhanced vision capabilities
                temperature=temperature,
                openai_api_key=settings.OPENAI_API_KEY,
                max_tokens=max_tokens,
                request_timeout=90,  # Increased timeout for GPT-5 vision processing
                verbosity=verbosity,  # Control response length and detail
                model_kwargs={
                    "response_format": {
                        "type": "json_object"
                    },  # Force JSON output for document analysis
                },
            )

    @staticmethod
    def get_guardrail_model(
        temperature: float = 0.1, max_tokens: int = 500, verbosity: str = "low"
    ) -> ChatOpenAI:
        """Get configured guardrail model using GPT-5 Mini for cost efficiency"""
        if StudyGuruModels.USE_FALLBACK_MODELS:
            # Fallback to GPT-4o-mini for compatibility
            return ChatOpenAI(
                model="gpt-4o-mini",  # Fallback model
                temperature=temperature,
                openai_api_key=settings.OPENAI_API_KEY,
                max_tokens=max_tokens,
                request_timeout=15,
                model_kwargs={
                    "response_format": {"type": "json_object"}
                },  # Force JSON output
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
                # GPT-5 specific parameters for better JSON output
                model_kwargs={
                    "response_format": {"type": "json_object"},  # Force JSON output
                },
            )

    @staticmethod
    def get_complex_reasoning_model(
        temperature: float = 0.1, max_tokens: int = 5000, verbosity: str = "medium"
    ) -> ChatOpenAI:
        """Get configured model for complex reasoning tasks using GPT-5 with high reasoning effort"""
        if StudyGuruModels.USE_FALLBACK_MODELS:
            # Fallback to GPT-4o for compatibility
            return ChatOpenAI(
                model="gpt-4o",  # Fallback model
                temperature=temperature,
                openai_api_key=settings.OPENAI_API_KEY,
                max_tokens=max_tokens,
                request_timeout=60,
                model_kwargs={
                    "response_format": {
                        "type": "json_object"
                    }  # Force JSON output for MCQ generation
                },
            )
        else:
            # GPT-5 configuration
            return ChatOpenAI(
                model="gpt-5",  # GPT-5 for complex reasoning
                temperature=temperature,
                openai_api_key=settings.OPENAI_API_KEY,
                max_tokens=max_tokens,
                request_timeout=150,  # Increased timeout for complex reasoning with high effort
                # GPT-5 specific parameters for complex tasks
                reasoning_effort="high",  # Maximum reasoning depth for complex problems
                verbosity=verbosity,  # Medium verbosity for complex reasoning tasks
                model_kwargs={
                    "response_format": {
                        "type": "json_object"
                    },  # Force JSON output for MCQ generation
                },
            )

    @staticmethod
    def get_embeddings_model() -> OpenAIEmbeddings:
        """Get configured embeddings model"""
        return OpenAIEmbeddings(
            model="text-embedding-3-small", openai_api_key=settings.OPENAI_API_KEY
        )

    @staticmethod
    def get_title_model(
        temperature: float = 0.3, max_tokens: int = 100, verbosity: str = "low"
    ) -> ChatOpenAI:
        """Get configured title generation model using GPT-5 Mini for cost efficiency"""
        if StudyGuruModels.USE_FALLBACK_MODELS:
            # Fallback to GPT-4o-mini for compatibility
            return ChatOpenAI(
                model="gpt-4o-mini",  # Fallback model
                temperature=temperature,
                openai_api_key=settings.OPENAI_API_KEY,
                max_tokens=max_tokens,  # Very low token limit for cost efficiency
                request_timeout=10,  # Fast timeout for quick response
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
            )


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
        """Get collection configuration"""
        return {
            "collection_name": settings.ZILLIZ_COLLECTION,
            "dimension": 1536,  # text-embedding-3-small dimension
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
        model = StudyGuruModels.get_guardrail_model(temperature=0.1, max_tokens=300)
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
            model_kwargs={"response_format": {"type": "json_object"}},
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
            model_kwargs={"response_format": {"type": "json_object"}},
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
            model_kwargs={"response_format": {"type": "json_object"}},
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
