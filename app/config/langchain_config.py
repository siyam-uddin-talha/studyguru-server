"""
LangChain configuration for StudyGuru Pro
"""

import os
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from app.core.config import settings


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
          
        - "written": Only if it's pure explanatory text, essays, or single concept explanations
        - "other": For mixed content or unclear format

        IMPORTANT: Most educational exercises with sub-parts should be classified as "mcq" type, even if they don't have traditional A/B/C/D choices.

        1. First, detect the language of the content
        2. Carefully identify the content type using the criteria above
        3. Provide a short, descriptive title for the page/content  
        4. Provide a summary title that describes what you will help the user with
        5. Based on the question type:
           - If MCQ: Extract each question/exercise part as a separate question
             * If the document contains actual multiple choice options (like: A, B, C, D or a, b, c, d or 1, 2, 3, 4), include them in the "options" field
             * If the document does NOT contain multiple choice options, omit the "options" field entirely
             * For the "answer" field: provide the correct option letter if options exist, or provide the actual solution/answer if no options are given
           - If written: Provide organized explanatory content

        Respond in the detected language and format your response as JSON with this structure:
        {{
            "type": "mcq" or "written" or "other",
            "language": "detected language",
            "title": "short descriptive title for the content",
            "summary_title": "summary of how you will help the user",
            "_result": {{
                // For MCQ type - extract each sub-part as a question:
                "questions": [
                    {{
                        "question": "question text (e.g., 'Solve |1 - 2x| = 3')",
                        // Include "options" field ONLY if the document contains actual multiple choice options
                        // If no options exist, omit this field entirely
                        "options": {{"a": "option1", "b": "option2", "c": "option3", "d": "option4"}},
                        "answer": "correct option letter (if options exist) or actual solution (if no options)",
                        "explanation": "step-by-step solution or brief explanation"
                    }}
                ]
                // For written type:
                "content": "organized explanatory text as you would provide in a chat response"
            }}
        }}
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
        You are a content guardrail for an educational platform. Review all user inputs—including text and any attached images—and determine whether the request violates any of the following rules:

        STRICT VIOLATION RULES:
        1. REJECT any images of people's faces, portraits, selfies, or photographs of individuals (even if they appear to be studying)
        2. REJECT content related to adult, explicit, or inappropriate material
        3. REJECT requests for direct code generation (e.g., "write a Java function"), except when analyzing educational code problems
        4. REJECT any content that is clearly NOT educational, study, or research-related
        
        IMPORTANT: Be VERY PERMISSIVE with educational content. When in doubt, ACCEPT rather than reject.

        ACCEPT THESE EDUCATIONAL CONTENT TYPES:
        - Textbooks, workbooks, and study guides
        - Educational worksheets and practice problems
        - Mathematical equations, formulas, and problem sets
        - Science diagrams, charts, and educational illustrations
        - Handwritten notes and study materials (WITHOUT faces)
        - Exercise sheets with numbered problems
        - Academic papers and research documents
        - Educational quizzes and assessments
        - Study notes and summaries
        - Any content clearly related to learning and education

        IMPORTANT FOR IMAGES:
        - If an image contains a human face or portrait: REJECT with violation_type "non_educational_content"
        - If an image is a selfie or photo of a person: REJECT with violation_type "non_educational_content"  
        - If an image is clearly educational content (textbook pages, worksheets, math problems, diagrams): ACCEPT
        - If an image shows mathematical problems, equations, or educational exercises: ACCEPT
        - If an image contains educational text, problems, or study materials: ACCEPT

        EXAMPLES OF ACCEPTABLE CONTENT:
        - Math worksheets with problems like "Solve |1 - 2x| = 3"
        - Exercise sheets titled "Exercise 1.1" with numbered problems
        - Science diagrams and educational charts
        - Handwritten study notes (without faces)
        - Textbook pages and educational materials

        You must respond with valid JSON in this exact format:
        {{
            "is_violation": boolean,
            "violation_type": "non_educational_content" | "inappropriate_content" | "code_generation" | null,
            "reasoning": "clear explanation of why this was rejected or accepted"
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

FORMATTING GUIDELINES:
1. Use clear section headers with ### for main topics
2. For MCQ content:
   - Number each question (1., 2., etc.)
   - If options exist: List options clearly (A., B., C., D.)
   - If no options exist: Provide the solution directly
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

Respond in valid JSON format:
{{
    "key_facts": [
        "fact 1: specific information learned with context",
        "fact 2: important concept discussed with details",
        "fact 3: problem solved or explained with solution"
    ],
    "main_topics": ["topic1", "topic2", "topic3"],
    "semantic_summary": "A concise 2-3 sentence summary capturing the essence of the conversation and its educational value",
    "important_terms": ["term1", "term2", "term3"],
    "context_for_future": "What context would be most useful for understanding follow-up questions in this conversation",
    "question_numbers": [1, 2, 3],
    "learning_progress": "What the user has learned or is learning",
    "potential_follow_ups": ["follow-up question 1", "follow-up question 2"],
    "difficulty_level": "beginner|intermediate|advanced",
    "subject_area": "math|science|language|other"
}}
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

Respond in valid JSON format:
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


class StudyGuruModels:
    """Model configurations for StudyGuru"""

    @staticmethod
    def get_chat_model(temperature: float = 0.2, max_tokens: int = 800) -> ChatOpenAI:
        """Get configured chat model - optimized for speed"""
        return ChatOpenAI(
            model="gpt-4o",  # Faster and cheaper model
            temperature=temperature,
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=max_tokens,
            request_timeout=30,  # Add timeout for faster failure
        )

    @staticmethod
    def get_vision_model(temperature: float = 0.3, max_tokens: int = 800) -> ChatOpenAI:
        """Get configured vision model - optimized for speed"""
        return ChatOpenAI(
            model="gpt-4o",  # Use GPT-4o for vision (faster than GPT-5)
            temperature=temperature,
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=max_tokens,
            request_timeout=45,  # Vision processing takes longer
        )

    @staticmethod
    def get_guardrail_model(
        temperature: float = 0.1, max_tokens: int = 300
    ) -> ChatOpenAI:
        """Get configured guardrail model (cost-optimized and fast)"""
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=max_tokens,
            request_timeout=15,  # Fast timeout for guardrails
        )

    @staticmethod
    def get_embeddings_model() -> OpenAIEmbeddings:
        """Get configured embeddings model"""
        return OpenAIEmbeddings(
            model="text-embedding-3-small", openai_api_key=settings.OPENAI_API_KEY
        )

    @staticmethod
    def get_title_model(temperature: float = 0.3, max_tokens: int = 100) -> ChatOpenAI:
        """Get configured title generation model (ultra cost-optimized)"""
        return ChatOpenAI(
            model="gpt-4o-mini",  # Most cost-effective model
            temperature=temperature,
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=max_tokens,  # Very low token limit for cost efficiency
            request_timeout=10,  # Fast timeout for quick response
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
    """Pre-configured chains for StudyGuru operations"""

    @staticmethod
    def get_document_analysis_chain():
        """Get document analysis chain"""
        model = StudyGuruModels.get_vision_model()
        parser = JsonOutputParser()
        return StudyGuruPrompts.DOCUMENT_ANALYSIS | model | parser

    @staticmethod
    def get_guardrail_chain():
        """Get guardrail check chain"""
        model = StudyGuruModels.get_guardrail_model(temperature=0.1, max_tokens=300)
        parser = JsonOutputParser()
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
        """Get title generation chain (cost-optimized)"""
        model = StudyGuruModels.get_title_model(temperature=0.3, max_tokens=100)
        parser = JsonOutputParser()
        return StudyGuruPrompts.TITLE_GENERATION | model | parser

    @staticmethod
    def get_conversation_summarization_chain():
        """Get conversation summarization chain"""
        model = StudyGuruModels.get_title_model(temperature=0.2, max_tokens=400)
        parser = JsonOutputParser()
        return StudyGuruPrompts.CONVERSATION_SUMMARIZATION | model | parser

    @staticmethod
    def get_interaction_summary_update_chain():
        """Get interaction summary update chain"""
        model = StudyGuruModels.get_title_model(temperature=0.2, max_tokens=500)
        parser = JsonOutputParser()
        return StudyGuruPrompts.INTERACTION_SUMMARY_UPDATE | model | parser


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

    # Default settings
    DEFAULT_MAX_TOKENS = 1000
    DEFAULT_TEMPERATURE = 0.2
    DEFAULT_TOP_K = 5

    # Points calculation
    POINTS_PER_TOKEN = 100

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
