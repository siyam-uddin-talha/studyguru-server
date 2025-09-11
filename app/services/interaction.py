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
from app.models.media import Media
from app.models.subscription import PointTransaction
from app.services.langchain_service import langchain_service
import json
from app.core.config import settings
from pydantic import BaseModel
from app.api.websocket_routes import notify_message_received
from app.api.sse_routes import notify_message_received_sse


# Guardrail agent is now handled by LangChain service


async def process_document_analysis(
    interaction_id: str, file_url: str, max_tokens: int
):
    """Background task to analyze a document/image URL with OpenAI Vision and persist results."""
    from app.core.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        try:
            # Get Interaction
            result = await db.execute(
                select(Interaction).where(Interaction.id == interaction_id)
            )
            interaction = result.scalar_one()

            # Get user
            user_result = await db.execute(
                select(User).where(User.id == interaction.user_id)
            )
            user = user_result.scalar_one()

            # Analyze with LangChain Vision using URL
            analysis_result = await langchain_service.analyze_document(
                file_url=file_url, max_tokens=max_tokens
            )

            tokens_used = analysis_result.get("token", 0)
            points_cost = langchain_service.calculate_points_cost(tokens_used)

            # Check if user still has enough points
            if user.current_points < points_cost:
                # Store failed AI conversation
                ai_failed = Conversation(
                    interaction_id=str(interaction.id),
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
                interaction_id=str(interaction.id),
                role=ConversationRole.AI,
                content=analysis_result,
                status="completed",
                tokens_used=int(tokens_used or 0),
                points_cost=int(points_cost or 0),
            )
            db.add(ai_conv)
            await db.commit()

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

            # Index into vector DB using LangChain
            try:
                content_text = None
                rtype = analysis_result.get("type")
                result_payload = analysis_result.get("result", {}) or {}
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
                    await langchain_service.upsert_embedding(
                        conv_id=str(ai_conv.id),
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
    interaction: Optional[Interaction],
    message: Optional[str],
    media_files: Optional[List[Dict[str, str]]],  # List of {id: str, url?: str}
    max_tokens: int = 1000,
) -> Dict[str, Any]:
    """
    Process a conversation turn: guardrails, retrieval, chat generation, points, embeddings.
    Returns a dict with {success, message, interaction_id}.
    """
    from app.core.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:

        interaction_id = interaction.id if interaction else None

        # Get media files if provided
        media_objects = []
        media_urls = []
        if media_files:
            for media_file in media_files:
                media_id = media_file.get("id")
                media_url = media_file.get("url")

                if media_url:
                    # If URL is provided, use it directly and create a mock media object
                    from app.models.media import Media

                    mock_media = Media(
                        id=media_id,
                        s3_key=media_url,  # Store URL in s3_key for compatibility
                        original_filename=f"media_{media_id}",
                        file_type="image/jpeg",  # Default type
                        original_size=0,
                    )
                    media_objects.append(mock_media)
                    media_urls.append(media_url)
                else:
                    # If no URL provided, query the database
                    media_result = await db.execute(
                        select(Media).where(Media.id == media_id)
                    )
                    media = media_result.scalar_one_or_none()
                    if media:
                        media_objects.append(media)
                        # Generate URL from S3 key
                        from app.core.config import settings

                        media_url = f"https://{settings.AWS_S3_BUCKET}.s3.amazonaws.com/{media.s3_key}"
                        media_urls.append(media_url)

        # Store user conversation
        user_conv = Conversation(
            interaction_id=str(interaction.id),
            role=ConversationRole.USER,
            content={
                "type": "text",
                "result": {"content": message or ""},
            },
            status="processing",
        )
        db.add(user_conv)
        await db.flush()  # Flush to get the ID but don't commit yet

        # Associate media files with conversation using junction table
        if media_objects:
            from app.models.interaction import conversation_files

            for media_obj in media_objects:
                # Insert directly into junction table
                await db.execute(
                    conversation_files.insert().values(
                        conversation_id=str(user_conv.id), media_id=str(media_obj.id)
                    )
                )

        # Commit the user conversation and media associations
        await db.commit()

        # Notify frontend that message was received
        try:
            # Try WebSocket first

            await notify_message_received(
                user_id=str(user_id),
                interaction_id=str(interaction.id),
                conversation_id=str(user_conv.id),
            )
        except Exception:
            try:
                # Fallback to SSE

                await notify_message_received_sse(
                    user_id=str(user_id),
                    interaction_id=str(interaction.id),
                    conversation_id=str(user_conv.id),
                )
            except Exception:
                # Don't fail the conversation if notifications fail
                pass

        # Guardrail check using LangChain
        try:
            if message or media_urls:
                guardrail_result = await langchain_service.check_guardrails(
                    message or "", media_urls
                )
                if guardrail_result.is_violation:
                    await db.commit()
                    return {
                        "success": False,
                        "message": f"Request blocked: {guardrail_result.violation_type}",
                        "interaction_id": str(interaction.id),
                        "ai_response": None,
                    }
        except Exception:
            pass

        # Enhanced Retrieval with RAG pattern
        context_text = ""
        try:
            # Build comprehensive query for vector search
            vector_query_parts = []

            # Add current message
            if message:
                vector_query_parts.append(message)

            # Add media descriptions if available
            if media_objects:
                for media in media_objects:
                    vector_query_parts.append(f"Document: {media.original_filename}")

            # For existing interactions, include interaction context
            if interaction.title:
                vector_query_parts.append(f"Topic: {interaction.title}")
            if interaction.summary_title:
                vector_query_parts.append(f"Context: {interaction.summary_title}")

            # Combine all parts for vector search
            vector_query = " ".join(vector_query_parts).strip()

            # Perform similarity search with higher top_k for better context
            similar = await langchain_service.similarity_search(
                query=(vector_query or "educational context"),
                user_id=str(user_id),
                top_k=8,  # Increased from 5 to 8 for better context
            )

            # Process and rank results
            context_parts: List[str] = []
            seen_titles = set()

            for m in similar or []:
                title = (m.get("title") or "").strip()
                content = (m.get("content") or "").strip()
                score = m.get("score", 0)
                metadata = m.get("metadata", {})

                # Skip if we've already included this title (avoid duplicates)
                if title in seen_titles:
                    continue
                seen_titles.add(title)

                # Only include high-quality matches (score threshold)
                if score > 0.3:  # Adjust threshold as needed
                    context_entry = []
                    if title and title not in ["User message", "AI response"]:
                        context_entry.append(f"**{title}**")
                    if content:
                        # Truncate very long content to keep context manageable
                        if len(content) > 500:
                            content = content[:500] + "..."
                        context_entry.append(content)

                    if context_entry:
                        context_parts.append("\n".join(context_entry))

            # Limit total context length to prevent token overflow
            context_text = "\n\n---\n\n".join(
                context_parts[:5]
            )  # Limit to top 5 results

        except Exception as e:
            # Log error but don't fail the conversation
            print(f"Vector search error: {e}")
            pass

        # Moderation fallback - using LangChain's built-in moderation
        try:
            if message:
                # LangChain handles moderation through the LLM's built-in safety features
                # Additional moderation can be added here if needed
                pass
        except Exception:
            pass

        # Chat generation using LangChain
        try:
            # Generate response using LangChain
            content_text, input_tokens, output_tokens, tokens_used = (
                await langchain_service.generate_conversation_response(
                    message=message or "",
                    context=context_text,
                    image_urls=media_urls if media_urls else None,
                    interaction_title=interaction.title,
                    interaction_summary=interaction.summary_title,
                    max_tokens=max_tokens,
                )
            )

            # For fresh interactions, try to extract title and summary_title from AI response
            if not interaction.title and not interaction.summary_title:
                try:
                    # Try to parse the response as JSON to extract structured data
                    parsed_response = json.loads(content_text)
                    if isinstance(parsed_response, dict):
                        extracted_title = parsed_response.get("title")
                        extracted_summary_title = parsed_response.get("summary_title")

                        if extracted_title:
                            interaction.title = extracted_title
                        if extracted_summary_title:
                            interaction.summary_title = extracted_summary_title

                        # Update the content_text to use the result field if available
                        if "result" in parsed_response:
                            result_content = parsed_response["result"]
                            if isinstance(result_content, dict):
                                if "content" in result_content:
                                    content_text = result_content["content"]
                                elif "questions" in result_content:
                                    # For MCQ, format the questions nicely
                                    questions = result_content["questions"]
                                    formatted_questions = []
                                    for q in questions:
                                        question_text = q.get("question", "")
                                        options = q.get("options", {})
                                        answer = q.get("answer", "")
                                        explanation = q.get("explanation", "")

                                        formatted_q = f"**{question_text}**\n"
                                        for opt_key, opt_value in options.items():
                                            formatted_q += f"{opt_key}. {opt_value}\n"
                                        if answer:
                                            formatted_q += f"**Answer:** {answer}\n"
                                        if explanation:
                                            formatted_q += (
                                                f"**Explanation:** {explanation}\n"
                                            )
                                        formatted_questions.append(formatted_q)

                                    content_text = "\n\n".join(formatted_questions)
                except (json.JSONDecodeError, KeyError, TypeError):
                    # If parsing fails, use the original content_text
                    pass

        except Exception as e:

            ai_failed = Conversation(
                interaction_id=str(interaction.id),
                role=ConversationRole.AI,
                content={"type": "other", "result": {"error": str(e)}},
                status="failed",
                error_message=str(e),
            )
            db.add(ai_failed)
            await db.commit()
            return {
                "success": False,
                "message": "AI generation failed",
                "ai_response": None,
            }

        # Points
        points_cost = langchain_service.calculate_points_cost(tokens_used)
        user = (await db.execute(select(User).where(User.id == user_id))).scalar_one()
        if user.current_points < points_cost:
            # Store the AI response content even with insufficient points
            # Extract the type from the AI response if it's structured
            ai_content_type = "other"  # default
            ai_result_content = content_text

            try:
                # Try to parse the AI response to extract the type
                if content_text:
                    # Check if the response is JSON structured
                    if content_text.strip().startswith("{"):
                        parsed_response = json.loads(content_text)
                        if isinstance(parsed_response, dict):
                            ai_content_type = parsed_response.get("type", "other")
                            # Extract the actual content from the result field
                            result_content = parsed_response.get("result", {})
                            if isinstance(result_content, dict):
                                if "content" in result_content:
                                    ai_result_content = result_content["content"]
                                elif "questions" in result_content:
                                    # For MCQ, format the questions
                                    questions = result_content["questions"]
                                    formatted_questions = []
                                    for q in questions:
                                        question_text = q.get("question", "")
                                        options = q.get("options", {})
                                        answer = q.get("answer", "")
                                        explanation = q.get("explanation", "")

                                        formatted_q = f"**{question_text}**\n"
                                        for opt_key, opt_value in options.items():
                                            formatted_q += f"{opt_key}. {opt_value}\n"
                                        if answer:
                                            formatted_q += f"**Answer:** {answer}\n"
                                        if explanation:
                                            formatted_q += (
                                                f"**Explanation:** {explanation}\n"
                                            )
                                        formatted_questions.append(formatted_q)

                                    ai_result_content = "\n\n".join(formatted_questions)
            except (json.JSONDecodeError, KeyError, TypeError):
                # If parsing fails, use the original content
                pass

            ai_failed = Conversation(
                interaction_id=str(interaction.id),
                role=ConversationRole.AI,
                content={
                    "type": ai_content_type,
                    "result": {
                        "content": ai_result_content,
                        "error": "Insufficient points for AI response",
                    },
                },
                status="failed",
                error_message="Insufficient points",
                tokens_used=tokens_used,
                points_cost=points_cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                is_hidden=True,
            )
            db.add(ai_failed)
            await db.commit()
            return {
                "success": False,
                "message": "Insufficient points",
                "interaction_id": str(interaction.id),
                # "ai_response": ai_result_content,  # Return the AI response content
            }

        user.current_points -= points_cost
        user.total_points_used += points_cost

        # Extract the type from the AI response if it's structured
        ai_content_type = "written"  # default
        ai_result_content = content_text

        try:
            # Try to parse the AI response to extract the type
            if content_text:
                # Check if the response is JSON structured
                if content_text.strip().startswith("{"):
                    parsed_response = json.loads(content_text)
                    if isinstance(parsed_response, dict):
                        ai_content_type = parsed_response.get("type", "written")
                        # Extract the actual content from the result field
                        result_content = parsed_response.get("result", {})
                        if isinstance(result_content, dict):
                            if "content" in result_content:
                                ai_result_content = result_content["content"]
                            elif "questions" in result_content:
                                # For MCQ, format the questions
                                questions = result_content["questions"]
                                formatted_questions = []
                                for q in questions:
                                    question_text = q.get("question", "")
                                    options = q.get("options", {})
                                    answer = q.get("answer", "")
                                    explanation = q.get("explanation", "")

                                    formatted_q = f"**{question_text}**\n"
                                    for opt_key, opt_value in options.items():
                                        formatted_q += f"{opt_key}. {opt_value}\n"
                                    if answer:
                                        formatted_q += f"**Answer:** {answer}\n"
                                    if explanation:
                                        formatted_q += (
                                            f"**Explanation:** {explanation}\n"
                                        )
                                    formatted_questions.append(formatted_q)

                                ai_result_content = "\n\n".join(formatted_questions)
        except (json.JSONDecodeError, KeyError, TypeError):
            # If parsing fails, use the original content
            pass

        ai_conv = Conversation(
            interaction_id=str(interaction.id),
            role=ConversationRole.AI,
            content={"type": ai_content_type, "result": {"content": ai_result_content}},
            status="completed",
            tokens_used=tokens_used,
            points_cost=points_cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        db.add(ai_conv)
        await db.commit()

        pt = PointTransaction(
            user_id=str(user_id),
            transaction_type="used",
            points=points_cost,
            description=f"Chat response - {tokens_used} tokens",
            conversation_id=str(ai_conv.id),
        )
        db.add(pt)

        # Embeddings using LangChain
        try:
            if message:
                await langchain_service.upsert_embedding(
                    conv_id=str(user_conv.id),
                    user_id=str(user_id),
                    text=message,
                    title="User message",
                    metadata={
                        "interaction_id": str(interaction.id),
                        "type": "user_message",
                        "has_media": len(media_objects) > 0,
                    },
                )
            if ai_result_content:
                await langchain_service.upsert_embedding(
                    conv_id=str(ai_conv.id),
                    user_id=str(user_id),
                    text=ai_result_content,
                    title="AI response",
                    metadata={
                        "interaction_id": str(interaction.id),
                        "type": "ai_response",
                    },
                )
        except Exception:
            pass

        await db.commit()

        # Notify frontend that AI response is ready
        try:
            # Try WebSocket first
            from app.api.websocket_routes import notify_ai_response_ready

            # await notify_ai_response_ready(
            #     user_id=str(user_id),
            #     interaction_id=str(interaction.id),
            #     ai_response=content_text,
            # )
        except Exception:
            try:
                # Fallback to SSE
                from app.api.sse_routes import notify_ai_response_ready_sse

                # await notify_ai_response_ready_sse(
                #     user_id=str(user_id),
                #     interaction_id=str(interaction.id),
                #     ai_response=content_text,
                # )
            except Exception:
                # Don't fail the conversation if notifications fail
                pass

        return {
            "success": True,
            "message": "Conversation updated",
            "interaction_id": str(interaction.id),
            "ai_response": ai_result_content,  # Return the processed AI response content
        }
