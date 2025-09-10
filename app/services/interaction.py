from fastapi import (
    APIRouter,
    Request,
    HTTPException,
    Depends,
)
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, List, Dict, Any
from app.models.user import User
from app.models.interaction import Interaction, Conversation, ConversationRole
from app.models.subscription import PointTransaction
from app.services.openai_service import OpenAIService
import json
import openai
from openai import OpenAI
from app.core.config import settings
from pydantic import BaseModel

openai.api_key = settings.OPENAI_API_KEY
client = OpenAI(api_key=settings.OPENAI_API_KEY)


class GuardrailOutput(BaseModel):
    is_violation: bool
    violation_type: Optional[str] = None
    reasoning: str


class GuardrailAgent:
    def __init__(self):
        self.name = "Guardrail check"
        self.instructions = """
        You are required to review all user inputs—including text and any attached images—and determine whether the request violates any of the following rules:

        1. Do not fulfill requests that ask for direct code generation (e.g., "write a Java function"), except when the user is presenting a question from an educational or research context that requires analysis or explanation.
        2. Prohibit content related to adult, explicit, or inappropriate material.
        3. Ensure all requests are strictly for educational, study, or research purposes. Any request outside this scope must be flagged as a violation.

        Provide a structured response indicating whether a violation has occurred. Include a clear boolean flag for violation and, if applicable, a brief explanation of the reason.
        """
        self.output_type = GuardrailOutput

    async def check(
        self, message: str, image_urls: Optional[List[str]] = None
    ) -> GuardrailOutput:
        """Run guardrail check using gpt-5-mini"""
        try:
            # Build input for guardrail check
            guardrail_input = []

            # System message
            guardrail_input.append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": self.instructions}],
                }
            )

            # User content with text and images
            user_content = []
            if message:
                user_content.append(
                    {"type": "input_text", "text": f"User message: {message}"}
                )

            if image_urls:
                for url in image_urls:
                    user_content.append({"type": "input_image", "image_url": url})

            if not user_content:
                user_content.append(
                    {"type": "input_text", "text": "No content provided"}
                )

            guardrail_input.append({"role": "user", "content": user_content})

            # Call gpt-5-mini for guardrail check
            response = client.responses.create(
                model="gpt-5-mini",
                input=guardrail_input,
                max_output_tokens=200,
                temperature=0.0,
            )

            # Parse response and create structured output
            response_text = response.output_text or ""

            # Simple parsing - look for violation indicators
            is_violation = any(
                word in response_text.lower()
                for word in [
                    "violation",
                    "violates",
                    "inappropriate",
                    "adult",
                    "not study",
                    "code writing",
                ]
            )

            violation_type = None
            if is_violation:
                if (
                    "code" in response_text.lower()
                    and "writing" in response_text.lower()
                ):
                    violation_type = "direct_code_writing"
                elif "adult" in response_text.lower():
                    violation_type = "adult_content"
                elif (
                    "study" in response_text.lower() and "not" in response_text.lower()
                ):
                    violation_type = "not_study_purpose"
                else:
                    violation_type = "general_violation"

            return GuardrailOutput(
                is_violation=is_violation,
                violation_type=violation_type,
                reasoning=response_text,
            )

        except Exception as e:
            # If guardrail fails, default to no violation but log the error
            return GuardrailOutput(
                is_violation=False,
                violation_type=None,
                reasoning=f"Guardrail check failed: {str(e)}",
            )


guardrail_agent = GuardrailAgent()


async def process_document_analysis(
    interaction_id: str, file_url: str, max_tokens: int
):
    """Background task to analyze a document/image URL with OpenAI Vision and persist results."""
    from app.core.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        try:
            # Get doc material
            result = await db.execute(
                select(Interaction).where(Interaction.id == interaction_id)
            )
            interaction = result.scalar_one()

            # Get user
            user_result = await db.execute(
                select(User).where(User.id == interaction.user_id)
            )
            user = user_result.scalar_one()

            # Analyze with OpenAI Vision using URL
            analysis_result = await OpenAIService.analyze_document(
                file_url=file_url, max_tokens=max_tokens
            )

            tokens_used = analysis_result.get("token", 0)
            points_cost = OpenAIService.calculate_points_cost(tokens_used)

            # Check if user still has enough points
            if user.current_points < points_cost:
                # Store failed AI conversation
                ai_failed = Conversation(
                    interaction_id=str(user.id),
                    role=ConversationRole.AI,
                    content=analysis_result,
                    status="failed",
                    error_message="Insufficient points",
                    tokens_used=int(tokens_used or 0),
                    points_cost=int(points_cost or 0),
                )
                db.add(ai_failed)
                await db.commit()
                return

            # Deduct points from user
            user.current_points -= points_cost
            user.total_points_used += points_cost

            # Persist AI conversation
            ai_conv = Conversation(
                interaction_id=str(user.id),
                role=ConversationRole.AI,
                content=analysis_result,
                status="completed",
                tokens_used=int(tokens_used or 0),
                points_cost=int(points_cost or 0),
            )
            db.add(ai_conv)
            await db.flush()

            # Create point transaction
            point_transaction = PointTransaction(
                user_id=user.id,
                transaction_type="used",
                points=int(points_cost or 0),
                description=f"Document analysis - {tokens_used} tokens",
                conversation_id=str(ai_conv.id),
            )
            db.add(point_transaction)

            # Update doc material
            interaction.analysis_response = analysis_result
            interaction.question_type = analysis_result.get("type")
            interaction.detected_language = analysis_result.get("language")
            interaction.title = analysis_result.get("title")
            interaction.summary_title = analysis_result.get("summary_title")
            interaction.tokens_used = tokens_used
            interaction.points_cost = points_cost
            interaction.status = "completed"

            # Optionally index into vector DB (Zilliz/Milvus)
            try:
                content_text = None
                rtype = analysis_result.get("type")
                result_payload = analysis_result.get("_result", {}) or {}
                if rtype == "written":
                    content_text = result_payload.get("content")
                elif rtype == "mcq":
                    # Concatenate questions for a coarse embedding
                    questions = result_payload.get("questions", []) or []
                    parts = []
                    for q in questions:
                        parts.append(str(q.get("question", "")))
                        opts = q.get("options", {}) or {}
                        if isinstance(opts, dict):
                            parts.extend([str(v) for v in opts.values()])
                    content_text = "\n".join(parts) if parts else None
                else:
                    content_text = result_payload.get("content") or json.dumps(
                        result_payload
                    )

                if content_text:
                    await OpenAIService.upsert_embedding(
                        doc_id=str(ai_conv.id),
                        user_id=str(user.id),
                        text=content_text,
                        title=interaction.title or interaction.summary_title,
                        metadata={
                            "question_type": rtype,
                            "language": analysis_result.get("language"),
                            "interaction_id": str(interaction.id),
                        },
                    )
            except Exception:
                # Do not fail the flow if vector DB is not available
                pass

            await db.commit()

        except Exception as e:
            # Update status to failed
            try:
                interaction.status = "failed"
                interaction.error_message = str(e)
                await db.commit()
            except Exception:
                pass


async def process_conversation_message(
    *,
    user_id: str,
    interaction_id: Optional[str],
    message: Optional[str],
    image_urls: Optional[List[str]],
    max_tokens: int = 1000,
) -> Dict[str, Any]:
    """
    Process a conversation turn: guardrails, retrieval, chat generation, points, embeddings.
    Returns a dict with {success, message, interaction_id}.
    """
    from app.core.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        # Fetch or create interaction
        interaction: Optional[Interaction] = None
        if interaction_id:
            result = await db.execute(
                select(Interaction).where(Interaction.id == interaction_id)
            )
            interaction = result.scalar_one_or_none()
            if not interaction or str(interaction.user_id) != str(user_id):
                return {"success": False, "message": "Interaction not found"}
        else:
            interaction = Interaction(user_id=str(user_id))
            db.add(interaction)
            await db.flush()

        # Store user conversation
        user_conv = Conversation(
            interaction_id=str(user_id),
            role=ConversationRole.USER,
            content={
                "type": "text",
                "_result": {"content": message or "", "images": image_urls or []},
            },
            status="completed",
        )
        db.add(user_conv)
        await db.flush()

        # Guardrail check using Agent approach
        try:
            if message or image_urls:
                guardrail_result = await guardrail_agent.check(
                    message or "", image_urls
                )
                if guardrail_result.is_violation:
                    await db.commit()
                    return {
                        "success": False,
                        "message": f"Request blocked: {guardrail_result.violation_type}",
                        "interaction_id": str(interaction.id),
                    }
        except Exception:
            pass

        # Retrieval
        context_text = ""
        try:
            vector_query = (message or "") + (
                "\n" + "\n".join(image_urls or []) if image_urls else ""
            )
            similar = await OpenAIService.similarity_search(
                query=(vector_query.strip() or "context"),
                top_k=5,
                user_id=str(user_id),
            )
            parts: List[str] = []
            for m in similar or []:
                t = (m.get("title") or "").strip()
                c = (m.get("content") or "").strip()
                if t:
                    parts.append(f"Title: {t}")
                if c:
                    parts.append(c)
            context_text = "\n\n".join(parts) if parts else ""
        except Exception:
            pass

        # Moderation fallback
        try:
            if message:
                mod = await openai.Moderation.acreate(
                    model="omni-moderation-latest", input=message
                )
                if mod and mod["results"][0]["flagged"]:
                    await db.commit()
                    return {
                        "success": False,
                        "message": "Your message violates our safety policy.",
                        "interaction_id": str(interaction.id),
                    }
        except Exception:
            pass

        # Build multi-modal user content
        # Reuse the analysis prompt from analyze_document for consistent image/document handling
        analysis_prompt = """
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
        system_prompt = analysis_prompt
        user_header = (
            f"Context (may be relevant):\n{context_text}\n\n" if context_text else ""
        ) + ("User: " + (message or "") if (message or "") else "")
        message_content: List[Dict[str, Any]] = []
        if user_header:
            message_content.append({"type": "text", "text": user_header})
        for url in image_urls or []:
            message_content.append({"type": "image_url", "image_url": {"url": url}})

        # Chat generation

        try:
            # Convert multi-modal message to Responses API input
            responses_input: List[Dict[str, Any]] = []
            sys_chunk = {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            }
            user_chunks: List[Dict[str, Any]] = []
            if message_content:
                for part in message_content:
                    if part.get("type") == "text":
                        user_chunks.append(
                            {"type": "input_text", "text": part.get("text", "")}
                        )
                    elif part.get("type") == "image_url":
                        url = (part.get("image_url") or {}).get("url")
                        if url:
                            user_chunks.append(
                                {"type": "input_image", "image_url": url}
                            )
            responses_input.append(sys_chunk)
            responses_input.append(
                {
                    "role": "user",
                    "content": user_chunks or [{"type": "input_text", "text": ""}],
                }
            )

            response = client.responses.create(
                model="gpt-5",
                input=responses_input,
                max_output_tokens=max_tokens or 500,
                temperature=0.2,
            )
            content_text = response.output_text or ""
            tokens_used = int(
                getattr(getattr(response, "usage", None), "total_tokens", 0) or 0
            )
        except Exception as e:
            ai_failed = Conversation(
                interaction_id=str(user_id),
                role=ConversationRole.AI,
                content={"type": "other", "_result": {"error": str(e)}},
                status="failed",
                error_message=str(e),
            )
            db.add(ai_failed)
            await db.commit()
            return {"success": False, "message": "AI generation failed"}

        # Points
        points_cost = OpenAIService.calculate_points_cost(tokens_used)
        user = (await db.execute(select(User).where(User.id == user_id))).scalar_one()
        if user.current_points < points_cost:
            ai_failed = Conversation(
                interaction_id=str(user_id),
                role=ConversationRole.AI,
                content={
                    "type": "other",
                    "_result": {"error": "Insufficient points for AI response"},
                },
                status="failed",
                error_message="Insufficient points",
                tokens_used=tokens_used,
                points_cost=points_cost,
            )
            db.add(ai_failed)
            await db.commit()
            return {"success": False, "message": "Insufficient points"}

        user.current_points -= points_cost
        user.total_points_used += points_cost

        ai_conv = Conversation(
            interaction_id=str(user_id),
            role=ConversationRole.AI,
            content={"type": "written", "_result": {"content": content_text}},
            status="completed",
            tokens_used=tokens_used,
            points_cost=points_cost,
        )
        db.add(ai_conv)
        await db.flush()

        pt = PointTransaction(
            user_id=str(user_id),
            transaction_type="used",
            points=points_cost,
            description=f"Chat response - {tokens_used} tokens",
            conversation_id=str(ai_conv.id),
        )
        db.add(pt)

        # Embeddings
        try:
            if message:
                await OpenAIService.upsert_embedding(
                    doc_id=str(user_conv.id),
                    user_id=str(user_id),
                    text=message,
                    title="User message",
                    metadata={
                        "interaction_id": str(interaction.id),
                        "type": "user_message",
                        "images": image_urls or [],
                    },
                )
            if content_text:
                await OpenAIService.upsert_embedding(
                    doc_id=str(ai_conv.id),
                    user_id=str(user_id),
                    text=content_text,
                    title="AI response",
                    metadata={
                        "interaction_id": str(interaction.id),
                        "type": "ai_response",
                    },
                )
        except Exception:
            pass

        await db.commit()

        return {
            "success": True,
            "message": "Conversation updated",
            "interaction_id": str(interaction.id),
        }
