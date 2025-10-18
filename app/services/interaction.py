from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import re
from app.models.user import User
from app.models.interaction import Interaction, Conversation, ConversationRole
from app.models.media import Media
from app.models.subscription import PointTransaction
from app.services.langchain_service import langchain_service
from app.services.document_integration_service import DocumentIntegrationService
from app.config.langchain_config import StudyGuruConfig

from app.core.database import AsyncSessionLocal
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
import asyncio

# Global task tracking for AI generation cancellation
active_generation_tasks: Dict[str, asyncio.Task] = {}

# Initialize document integration service
document_integration_service = DocumentIntegrationService()


async def process_conversation_message(
    *,
    user: User,
    interaction: Optional[Interaction],
    message: Optional[str],
    media_files: Optional[List[Dict[str, str]]],
    max_tokens: Optional[int] = None,  # Will be calculated dynamically if not provided
    db: Optional[AsyncSession] = None,  # Database session parameter
) -> Dict[str, Any]:
    """
    Optimized conversation processor with parallel operations and aggressive caching.
    """

    # Validate input - ensure user has provided some content
    if not message or not message.strip():
        if not media_files or len(media_files) == 0:
            return {
                "success": False,
                "message": "Please provide a message or attach a file",
                "interaction_id": str(interaction.id) if interaction else None,
                "ai_response": "Please provide a message or attach a file to get help.",
            }

    # Calculate dynamic token limit based on file count
    if max_tokens is None:
        file_count = len(media_files) if media_files else 0
        max_tokens = StudyGuruConfig.calculate_dynamic_tokens(file_count)
        print(f"🔢 Dynamic token calculation: {file_count} files = {max_tokens} tokens")

    # Use provided DB session or create a new one
    should_close_db = False
    if db is None:
        db = AsyncSessionLocal()
        should_close_db = True

    try:
        user_id = user.id
        interaction_id = interaction.id if interaction else None

        # If interaction was passed from resolver, merge it into this session
        if interaction:
            interaction = await db.merge(interaction)

            # === PHASE 1: PARALLEL SETUP & MEDIA PROCESSING ===
            async def process_media_parallel():
                """Process media files in parallel"""
                if not media_files:
                    return [], []

                media_objects = []
                media_urls = []

                # Process all media files concurrently
                media_ids_to_fetch = []

                for media_file in media_files:
                    media_id = media_file.get("id")
                    media_url = media_file.get("url")

                    if media_url:
                        # Direct URL - create mock media object
                        mock_media = Media(
                            id=media_id,
                            s3_key=media_url,
                            original_filename=f"media_{media_id}",
                            file_type="image/jpeg",
                            original_size=0,
                        )
                        media_objects.append(mock_media)
                        media_urls.append(media_url)
                    else:
                        # Collect media IDs for batch fetch
                        media_ids_to_fetch.append(media_id)

                # Single optimized query for all media files by ID
                if media_ids_to_fetch:
                    result = await db.execute(
                        select(Media).where(Media.id.in_(media_ids_to_fetch))
                    )
                    fetched_media = result.scalars().all()

                    for media in fetched_media:
                        media_objects.append(media)
                        media_url = f"https://{settings.AWS_S3_BUCKET}.s3.amazonaws.com/{media.s3_key}"
                        media_urls.append(media_url)

                return media_objects, media_urls

            async def create_user_conversation():
                """Create user conversation record"""
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
                await db.flush()
                return user_conv

            # Run media processing and user conversation creation in parallel
            (media_objects, media_urls), user_conv = await asyncio.gather(
                process_media_parallel(), create_user_conversation()
            )

            # Associate media files if present (non-blocking)
            if media_objects:
                from app.models.interaction import conversation_files

                media_tasks = [
                    db.execute(
                        conversation_files.insert().values(
                            conversation_id=str(user_conv.id),
                            media_id=str(media_obj.id),
                        )
                    )
                    for media_obj in media_objects
                ]
                await asyncio.gather(*media_tasks, return_exceptions=True)

            # Commit user message immediately for instant notification
            await db.commit()

            print(f"\n🚀 USER MESSAGE STORED - Sending immediate notification")
            print(f"   User ID: {user_id}")
            print(f"   Interaction ID: {interaction.id}")
            print(f"   Conversation ID: {user_conv.id}\n")

            # Send notification IMMEDIATELY after user message is stored
            # This gives instant feedback to the user with typing indicator
            notification_task = asyncio.create_task(
                _send_notifications_background(
                    str(user_id), str(interaction.id), str(user_conv.id)
                )
            )
            # Don't await - let it run in background, but keep reference
            notification_task.add_done_callback(
                lambda t: (
                    print(f"✅ Message notification task completed")
                    if not t.exception()
                    else print(f"❌ Message notification task failed: {t.exception()}")
                )
            )

            # === PHASE 2: PARALLEL VALIDATION & CONTEXT RETRIEVAL ===
            import time

            phase2_start = time.time()
            print(
                f"🚀 PHASE 2 START: Guardrails + Context Retrieval at {time.time():.3f}"
            )

            async def run_guardrails():
                """Run guardrail checks"""
                guardrail_start = time.time()
                print(f"🛡️ GUARDRAIL START: {guardrail_start:.3f}")

                if not (message or media_urls):
                    print(f"🛡️ GUARDRAIL SKIP: No content to check")
                    return None
                try:
                    result = await langchain_service.check_guardrails(
                        message or "", media_urls
                    )
                    guardrail_end = time.time()
                    print(
                        f"🛡️ GUARDRAIL COMPLETE: {guardrail_end:.3f} (took {guardrail_end - guardrail_start:.3f}s)"
                    )
                    return result
                except Exception as e:
                    guardrail_end = time.time()
                    print(
                        f"🛡️ GUARDRAIL ERROR: {guardrail_end:.3f} (took {guardrail_end - guardrail_start:.3f}s) - {e}"
                    )
                    return None

            async def get_context_fast():
                """Enhanced context retrieval using multi-level strategy"""
                context_start = time.time()
                print(f"🔍 CONTEXT START: {context_start:.3f}")

                try:
                    from app.services.context_service import context_service

                    # Use the new enhanced context retrieval service
                    context_result = await context_service.get_comprehensive_context(
                        user_id=str(user_id),
                        interaction_id=str(interaction.id) if interaction else None,
                        message=message or "",
                        include_cross_interaction=True,
                        max_context_length=4000,
                    )

                    context_text = context_result.get("context", "")
                    metadata = context_result.get("metadata", {})

                    context_end = time.time()
                    print(
                        f"🔍 CONTEXT COMPLETE: {context_end:.3f} (took {context_end - context_start:.3f}s)"
                    )

                    print(f"🔍 Enhanced Context Retrieval Results:")
                    print(f"   Sources used: {metadata.get('sources_used', [])}")
                    print(f"   Sources ignored: {metadata.get('sources_ignored', [])}")
                    print(
                        f"   Total retrieval time: {metadata.get('total_retrieval_time', 0):.3f}s"
                    )
                    print(f"   Context length: {len(context_text)} characters")

                    # Log context usage for monitoring
                    if interaction:
                        await context_service.log_context_usage(
                            user_id=str(user_id),
                            interaction_id=str(interaction.id),
                            conversation_id=None,  # Will be set later
                            context_sources_used=metadata.get("sources_used", []),
                            context_sources_ignored=metadata.get("sources_ignored", []),
                            retrieval_time=metadata.get("total_retrieval_time", 0),
                            user_query=message or "",
                            query_type="general_inquiry",
                        )

                    return context_text
                except Exception as e:
                    context_end = time.time()
                    print(
                        f"🔍 CONTEXT ERROR: {context_end:.3f} (took {context_end - context_start:.3f}s) - {e}"
                    )
                    return ""

            # Run guardrails and context retrieval in parallel
            print(f"🔄 Starting parallel guardrails + context retrieval...")
            guardrail_result, context_text = await asyncio.gather(
                run_guardrails(), get_context_fast(), return_exceptions=True
            )

            phase2_end = time.time()
            print(
                f"✅ PHASE 2 COMPLETE: {phase2_end:.3f} (took {phase2_end - phase2_start:.3f}s)"
            )

            print(f"🔍 Context text: {context_text}")

            # Debug guardrail result
            if not isinstance(guardrail_result, Exception) and guardrail_result:
                print(f"🛡️ GUARDRAIL DEBUG - Violation: {guardrail_result.is_violation}")
                print(f"🛡️ GUARDRAIL DEBUG - Type: {guardrail_result.violation_type}")
                print(f"🛡️ GUARDRAIL DEBUG - Reasoning: {guardrail_result.reasoning}")
            else:
                print(f"🛡️ GUARDRAIL DEBUG - No violation detected")

            # Check if context retrieval failed
            if isinstance(context_text, Exception):
                print(f"⚠️ Context retrieval failed: {context_text}")
                context_text = ""
            elif not context_text:
                print(f"⚠️ No context retrieved")
            else:
                print(
                    f"✅ Context retrieved successfully: {len(context_text)} characters"
                )

            # Check if guardrail violation occurred
            if (
                not isinstance(guardrail_result, Exception)
                and guardrail_result
                and guardrail_result.is_violation
            ):
                print(f"🛡️ GUARDRAIL VIOLATION - Blocking request")
                return {
                    "success": False,
                    "message": "Content violates platform policies",
                    "violation_type": guardrail_result.violation_type,
                    "reasoning": guardrail_result.reasoning,
                }

            # Generate AI response
            print(f"🤖 Generating AI response...")
            print(f"   Message: {message[:100]}..." if message else "   No message")
            print(f"   Context length: {len(context_text)} characters")
            print(f"   Max tokens: {max_tokens}")
            print(f"   Media files: {len(media_urls) if media_urls else 0}")

            # Check if we have images that need document analysis
            if (
                media_urls
                and len(media_urls) > 0
                and settings.ENABLE_ENHANCED_PROCESSING
            ):
                doc_processing_start = time.time()
                print(f"📋 DOCUMENT PROCESSING START: {doc_processing_start:.3f}")
                print(
                    "📋 ENHANCED DOCUMENT PROCESSING MODE - Processing uploaded images"
                )

                # Use the new document integration service for comprehensive processing
                from app.services.document_integration_service import (
                    document_integration_service,
                )

                analysis_results = []
                total_tokens = 0
                processed_documents = []

                # Calculate tokens per attachment more accurately
                base_tokens = 2000
                attachment_tokens = (
                    max(500, (max_tokens - base_tokens) // len(media_urls))
                    if len(media_urls) > 0
                    else 1000
                )

                print(f"📊 Document processing config:")
                print(f"   Media URLs: {len(media_urls)}")
                print(f"   Attachment tokens: {attachment_tokens}")
                print(f"   Base tokens: {base_tokens}")
                print(f"   Max tokens: {max_tokens}")

                # Process all documents in parallel for much faster processing
                async def process_single_document(
                    i: int, media_url: str
                ) -> Dict[str, Any]:
                    """Process a single document with error handling"""
                    doc_start = time.time()
                    print(f"🔍 DOC {i+1} START: {doc_start:.3f} - {media_url}")
                    print(f"📊 Using {attachment_tokens} tokens for this document")

                    try:
                        # Get media object for this URL
                        media_obj = media_objects[i] if i < len(media_objects) else None
                        media_id = media_obj.id if media_obj else f"media_{i}"

                        # Use comprehensive document processing
                        analysis_start = time.time()
                        print(f"🔍 DOC {i+1} ANALYSIS START: {analysis_start:.3f}")

                        document_analysis = await document_integration_service.process_document_comprehensive(
                            media_id=media_id,
                            interaction_id=str(interaction.id),
                            user_id=str(user_id),
                            file_url=media_url,
                            max_tokens=attachment_tokens,
                        )

                        analysis_end = time.time()
                        print(
                            f"🔍 DOC {i+1} ANALYSIS COMPLETE: {analysis_end:.3f} (took {analysis_end - analysis_start:.3f}s)"
                        )

                        # Convert to expected format for backward compatibility
                        analysis_result = {
                            "type": document_analysis.document_type,
                            "language": document_analysis.metadata.get(
                                "language", "english"
                            ),
                            "title": document_analysis.metadata.get(
                                "title", f"Document {i+1}"
                            ),
                            "summary_title": document_analysis.content_summary,
                            "token": attachment_tokens,
                            "_result": {
                                "content": document_analysis.full_content,
                                "questions": [
                                    {
                                        "question": q.question_text,
                                        "options": (
                                            q.metadata.get("options", {})
                                            if q.metadata
                                            else {}
                                        ),
                                        "answer": (
                                            q.metadata.get("answer", "")
                                            if q.metadata
                                            else ""
                                        ),
                                        "explanation": (
                                            q.metadata.get("explanation", "")
                                            if q.metadata
                                            else ""
                                        ),
                                    }
                                    for q in document_analysis.chunks
                                    if q.chunk_type == "question"
                                ],
                                "document_analysis": {
                                    "total_questions": document_analysis.total_questions,
                                    "main_topics": document_analysis.main_topics,
                                    "difficulty_level": document_analysis.difficulty_level,
                                    "subject_area": document_analysis.subject_area,
                                    "key_concepts": document_analysis.key_concepts,
                                    "chunk_count": len(document_analysis.chunks),
                                },
                            },
                        }

                        doc_end = time.time()
                        print(
                            f"✅ DOC {i+1} COMPLETE: {doc_end:.3f} (took {doc_end - doc_start:.3f}s)"
                        )
                        print(f"   Type: {document_analysis.document_type}")
                        print(f"   Questions: {document_analysis.total_questions}")
                        print(f"   Topics: {document_analysis.main_topics}")
                        print(f"   Chunks: {len(document_analysis.chunks)}")

                        return analysis_result, document_analysis

                    except Exception as doc_error:
                        print(
                            f"❌ Document processing failed for {media_url}: {doc_error}"
                        )
                        # Fallback to basic analysis
                        try:
                            basic_analysis = await langchain_service.analyze_document(
                                file_url=media_url, max_tokens=attachment_tokens
                            )
                            if basic_analysis.get("type") != "error":
                                return basic_analysis, None
                        except Exception as basic_error:
                            print(f"❌ Basic analysis also failed: {basic_error}")

                        # Return error result
                        return {
                            "type": "error",
                            "language": "unknown",
                            "title": f"Document {i+1}",
                            "summary_title": "Processing failed",
                            "token": 0,
                            "_result": {
                                "error": f"Unable to process document: {str(doc_error)}",
                                "details": (
                                    str(basic_error)
                                    if "basic_error" in locals()
                                    else str(doc_error)
                                ),
                            },
                        }, None

                # Process all documents in parallel
                parallel_start = time.time()
                print(f"🚀 PARALLEL PROCESSING START: {parallel_start:.3f}")
                print(f"🚀 Processing {len(media_urls)} documents in parallel...")
                document_tasks = [
                    process_single_document(i, media_url)
                    for i, media_url in enumerate(media_urls)
                ]

                # Wait for all documents to complete
                print(f"⏳ Waiting for all documents to complete...")
                document_results = await asyncio.gather(
                    *document_tasks, return_exceptions=True
                )

                parallel_end = time.time()
                print(
                    f"✅ PARALLEL PROCESSING COMPLETE: {parallel_end:.3f} (took {parallel_end - parallel_start:.3f}s)"
                )

                # Process results
                for i, result in enumerate(document_results):
                    if isinstance(result, Exception):
                        print(f"❌ Document {i+1} failed with exception: {result}")
                        analysis_results.append(
                            {
                                "type": "error",
                                "language": "unknown",
                                "title": f"Document {i+1}",
                                "summary_title": "Processing failed",
                                "token": 0,
                                "_result": {"error": f"Exception: {str(result)}"},
                            }
                        )
                    else:
                        analysis_result, document_analysis = result
                        analysis_results.append(analysis_result)
                        if document_analysis:
                            processed_documents.append(document_analysis)
                        total_tokens += attachment_tokens

                doc_processing_end = time.time()
                print(
                    f"✅ DOCUMENT PROCESSING COMPLETE: {doc_processing_end:.3f} (took {doc_processing_end - doc_processing_start:.3f}s)"
                )

                if analysis_results:

                    print(f"📊 Total tokens used: {total_tokens}")

                    # Create conversation entry for document analysis
                    analysis_content = {
                        "type": "document_analysis",
                        "_result": {
                            "note": f"Processed {len(analysis_results)} document(s) with enhanced analysis",
                            "results": analysis_results,
                            "total_tokens": total_tokens,
                            "enhanced_processing": True,
                            "processed_documents": [
                                {
                                    "document_type": doc.document_type,
                                    "total_questions": doc.total_questions,
                                    "main_topics": doc.main_topics,
                                    "difficulty_level": doc.difficulty_level,
                                    "subject_area": doc.subject_area,
                                    "chunk_count": len(doc.chunks),
                                }
                                for doc in processed_documents
                            ],
                        },
                    }

                    # Create AI conversation entry
                    ai_conv = Conversation(
                        interaction_id=str(interaction.id),
                        role=ConversationRole.AI,
                        content=analysis_content,
                        input_tokens=total_tokens,
                        output_tokens=0,
                        tokens_used=total_tokens,
                        points_cost=langchain_service.calculate_points_cost(
                            total_tokens
                        ),
                        status="completed",
                    )

                    db.add(ai_conv)
                    await db.flush()

                    return {
                        "success": True,
                        "message": "Enhanced document processing completed",
                        "conversation_id": str(ai_conv.id),
                        "analysis_results": analysis_results,
                        "total_tokens": total_tokens,
                        "points_cost": ai_conv.points_cost,
                        "enhanced_processing": True,
                        "processed_documents": len(processed_documents),
                    }
                else:
                    return {
                        "success": False,
                        "message": "Document processing failed for all uploaded files",
                    }

            # Generate conversation response using LangChain
            ai_generation_start = time.time()
            print(f"🤖 AI GENERATION START: {ai_generation_start:.3f}")

            # Create a task for AI generation that can be cancelled
            task_key = f"{user_id}_{interaction.id}"

            async def generate_ai_response():
                ai_start = time.time()
                print(f"🤖 AI SERVICE CALL START: {ai_start:.3f}")

                result = await langchain_service.generate_conversation_response(
                    message=message or "",
                    context=context_text,
                    image_urls=media_urls,
                    interaction_title=interaction.title if interaction else None,
                    interaction_summary=(
                        interaction.semantic_summary.get("updated_summary")
                        if interaction and interaction.semantic_summary
                        else None
                    ),
                    max_tokens=max_tokens,
                )

                ai_end = time.time()
                print(
                    f"🤖 AI SERVICE CALL COMPLETE: {ai_end:.3f} (took {ai_end - ai_start:.3f}s)"
                )
                return result

            # Create and track the task
            generation_task = asyncio.create_task(generate_ai_response())
            active_generation_tasks[task_key] = generation_task

            try:
                print(f"⏳ Waiting for AI response...")
                ai_response, input_tokens, output_tokens, total_tokens = (
                    await generation_task
                )

                ai_generation_end = time.time()
                print(
                    f"🤖 AI GENERATION COMPLETE: {ai_generation_end:.3f} (took {ai_generation_end - ai_generation_start:.3f}s)"
                )
                print("=" * 200)
                print(f"🔄 AI RESPONSE TYPE: {type(ai_response)}")
                print(
                    f"🔄 AI RESPONSE LENGTH: {len(str(ai_response)) if ai_response else 0}"
                )
                print(f"🔄 AI RESPONSE: {ai_response}")
                print("=" * 200)

                # Check if AI response is empty or None
                if not ai_response or (
                    isinstance(ai_response, str) and ai_response.strip() == ""
                ):
                    print(f"❌ ERROR: AI response is empty or None!")
                    print(f"   Input tokens: {input_tokens}")
                    print(f"   Output tokens: {output_tokens}")
                    print(f"   Total tokens: {total_tokens}")
                    return {
                        "success": False,
                        "message": "AI response was empty",
                        "interaction_id": str(interaction.id),
                        "ai_response": "Sorry, I couldn't generate a response. Please try again.",
                    }

            except asyncio.CancelledError:

                # Clean up the task from tracking
                if task_key in active_generation_tasks:
                    del active_generation_tasks[task_key]
                return {
                    "success": False,
                    "message": "AI generation was cancelled",
                    "interaction_id": str(interaction.id),
                }
            except Exception as e:

                # Clean up the task from tracking
                if task_key in active_generation_tasks:
                    del active_generation_tasks[task_key]
                return {
                    "success": False,
                    "message": f"Failed to generate response: {str(e)}",
                }
            finally:
                # Clean up the task from tracking when done
                if task_key in active_generation_tasks:
                    del active_generation_tasks[task_key]

            # Create user conversation entry (store media IDs instead of URLs)
            user_conv = Conversation(
                interaction_id=str(interaction.id),
                role=ConversationRole.USER,
                content={
                    "type": "text",
                    "_result": {
                        "note": message or "",
                        "media_ids": (
                            [media.id for media in media_objects]
                            if media_objects
                            else []
                        ),
                    },
                },
                status="completed",
            )

            db.add(user_conv)
            await db.flush()

            # Send SSE notification that user message was received
            try:
                from app.api.sse_routes import notify_message_received_sse

                await notify_message_received_sse(
                    user_id=str(user_id),
                    interaction_id=str(interaction.id),
                    conversation_id=str(user_conv.id),
                )

            except Exception as e:
                print(f"⚠️  Failed to send message received notification: {e}")

            # Process AI content with enhanced formatting - with robust error handling
            print(
                f"🔧 CALLING _process_ai_content_fast with ai_response: {ai_response}"
            )

            # Initialize fallback values
            ai_content_type = "written"
            processed_ai_response = (
                str(ai_response) if ai_response else "No response generated"
            )

            try:
                ai_content_type, processed_ai_response = _process_ai_content_fast(
                    ai_response
                )
                print(
                    f"🔧 _process_ai_content_fast returned: type={ai_content_type}, content={processed_ai_response}"
                )
            except Exception as processing_error:
                print(f"⚠️ Error processing AI content: {processing_error}")
                print(f"🔧 Using fallback processing for AI response")
                # Use fallback processing - store raw response
                if isinstance(ai_response, dict):
                    ai_content_type = "structured"
                    processed_ai_response = str(ai_response)
                else:
                    ai_content_type = "written"
                    processed_ai_response = (
                        str(ai_response) if ai_response else "No response generated"
                    )
                print(
                    f"🔧 Fallback result: type={ai_content_type}, content={processed_ai_response}"
                )

            # Ensure we have valid content to store
            if not processed_ai_response or (
                isinstance(processed_ai_response, str)
                and processed_ai_response.strip() == ""
            ):
                processed_ai_response = (
                    "AI response was generated but could not be processed"
                )
                ai_content_type = "error"
                print(f"🔧 No valid content, using error fallback")

            # Create AI conversation entry with enhanced content
            ai_conv = Conversation(
                interaction_id=str(interaction.id),
                role=ConversationRole.AI,
                content={
                    "type": ai_content_type,
                    "_result": {"content": processed_ai_response},
                },
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tokens_used=total_tokens,
                points_cost=langchain_service.calculate_points_cost(total_tokens),
                status="completed",
            )

            # CRITICAL: Always store the AI response in database, regardless of other errors
            try:
                db.add(ai_conv)
                await db.flush()
                print(f"✅ AI conversation entry added to database successfully")
            except Exception as db_error:
                print(
                    f"❌ CRITICAL: Failed to add AI conversation to database: {db_error}"
                )
                # This is a critical error - we must not continue without storing the response
                raise Exception(f"Failed to store AI response in database: {db_error}")

            # Update interaction title if needed using enhanced metadata extraction
            try:
                await _extract_interaction_metadata_fast(
                    interaction=interaction,
                    content_text=ai_response,
                    original_message=message or "",
                )

            except Exception as title_error:
                print(f"⚠️ Metadata extraction failed: {title_error}")
                # Don't let title generation failure break the main AI response

            # CRITICAL: Always commit the AI response, even if other operations fail
            try:
                await db.commit()
                print(f"✅ AI response committed to database successfully")
            except Exception as commit_error:
                print(
                    f"❌ CRITICAL: Failed to commit AI response to database: {commit_error}"
                )
                # This is a critical error - we must not continue without committing the response
                raise Exception(
                    f"Failed to commit AI response to database: {commit_error}"
                )

            # Refresh the interaction to verify the changes were saved
            try:
                await db.refresh(interaction)
                print(f"✅ Interaction refreshed successfully")
            except Exception as refresh_error:
                print(f"⚠️ Failed to refresh interaction: {refresh_error}")
                # This is not critical - the AI response is already committed

            # Start background operations using real-time context service
            asyncio.create_task(
                _background_operations_enhanced(
                    user_conv.id,
                    ai_conv.id,
                    str(user_id),
                    str(interaction.id),
                    message or "",
                    ai_response,
                )
            )

            # Send AI response notification using enhanced function
            try:
                await _send_ai_response_notification(
                    user_id=str(user_id),
                    interaction_id=str(interaction.id),
                    ai_response=ai_response,
                )
                print(f"✅ Sent AI response notification")
            except Exception as e:
                print(f"⚠️  Failed to send AI response notification: {e}")

            result = {
                "success": True,
                "message": "Response generated successfully",
                "conversation_id": str(ai_conv.id),
                "ai_response": processed_ai_response,
                "ai_content_type": ai_content_type,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "points_cost": ai_conv.points_cost,
            }
            print(f"🔧 SERVICE RETURNING RESULT: {result}")
            return result

    except Exception as e:
        print(f"❌ Error in process_conversation_message: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "message": f"An error occurred: {str(e)}"}
    finally:
        # Close the database session if we created it
        if should_close_db and db:
            await db.close()


async def generate_mcq_questions(
    *,
    user: User,
    interaction: Optional[Interaction],
    topic_or_content: str,
    max_tokens: int = 1200,
) -> Dict[str, Any]:
    """
    Generate MCQ questions for a given topic or content
    """
    try:
        # Use a single DB session for the entire operation
        async with AsyncSessionLocal() as db:
            # Get or create interaction
            if not interaction:
                interaction = Interaction(
                    user_id=user.id,
                    title="MCQ Generation",
                    summary="Generated multiple choice questions",
                )
                db.add(interaction)
                await db.flush()

            # Generate MCQ questions using LangChain
            mcq_result = await langchain_service.generate_mcq_questions(
                topic_or_content=topic_or_content, max_tokens=max_tokens
            )

            if mcq_result.get("type") == "error":
                return {
                    "success": False,
                    "message": "Failed to generate MCQs",
                    "error": mcq_result.get("_result", {}).get(
                        "error", "Unknown error"
                    ),
                }

            # Create conversation entries
            user_conv = Conversation(
                interaction_id=interaction.id,
                role=ConversationRole.USER,
                content=f"Generate MCQs for: {topic_or_content}",
                points_cost=0,
            )
            db.add(user_conv)

            # Format the MCQ response for display
            questions = mcq_result.get("_result", {}).get("questions", [])
            formatted_response = f"# {mcq_result.get('title', 'MCQ Questions')}\n\n"

            for i, question in enumerate(questions, 1):
                formatted_response += f"**{i}.** {question.get('question', '')}\n\n"

                options = question.get("options", {})
                if options:
                    for key, value in options.items():
                        formatted_response += f"   {key.upper()}. {value}\n"
                    formatted_response += (
                        f"\n**Answer:** {question.get('answer', '').upper()}\n"
                    )
                else:
                    formatted_response += f"**Answer:** {question.get('answer', '')}\n"

                formatted_response += (
                    f"**Explanation:** {question.get('explanation', '')}\n\n"
                )

            ai_conv = Conversation(
                interaction_id=interaction.id,
                role=ConversationRole.AI,
                content=formatted_response,
                points_cost=StudyGuruConfig.calculate_points_cost(
                    mcq_result.get("token", 0)
                ),
            )
            db.add(ai_conv)

            # Update interaction title and summary
            interaction.title = mcq_result.get("title", "MCQ Generation")
            interaction.summary = mcq_result.get(
                "summary_title", "Generated multiple choice questions"
            )

            # Commit all changes
            await db.commit()

            return {
                "success": True,
                "message": "MCQ questions generated successfully",
                "conversation_id": str(ai_conv.id),
                "ai_response": formatted_response,
                "mcq_data": mcq_result,
                "total_questions": len(questions),
                "points_cost": ai_conv.points_cost,
            }

    except Exception as e:
        print(f"❌ Error in generate_mcq_questions: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "message": f"An error occurred: {str(e)}"}


async def _background_operations_enhanced(
    user_conv_id: int,
    ai_conv_id: int,
    user_id: str,
    interaction_id: str,
    message: str,
    ai_content: str,
):
    """Enhanced background operations using real-time context service"""
    try:
        print(
            f"🔄 Starting enhanced background operations for interaction {interaction_id}"
        )

        # Import the real-time context service
        from app.services.real_time_context_service import real_time_context_service

        # === STEP 1: Queue semantic summary update ===
        semantic_task_id = await real_time_context_service.queue_context_update(
            user_id=user_id,
            interaction_id=interaction_id,
            conversation_id=str(ai_conv_id),
            update_type="semantic_summary",
            payload={"user_message": message, "ai_response": ai_content},
            priority=1,  # High priority
        )
        print(f"   📝 Queued semantic summary update: {semantic_task_id}")

        # === STEP 2: Queue embedding updates ===
        embedding_tasks = []

        # Prepare enhanced metadata
        enhanced_metadata = {
            "interaction_id": interaction_id,
            "conversation_id": str(ai_conv_id),
            "update_timestamp": datetime.now().isoformat(),
            "context_version": "2.0",
        }

        # Queue user message embedding
        if message:
            user_embedding_task_id = (
                await real_time_context_service.queue_context_update(
                    user_id=user_id,
                    interaction_id=interaction_id,
                    conversation_id=str(user_conv_id),
                    update_type="embedding",
                    payload={
                        "conversation_id": str(user_conv_id),
                        "text": message,
                        "title": f"User message in {interaction_id}",
                        "metadata": enhanced_metadata,
                    },
                    priority=2,
                )
            )
            embedding_tasks.append(user_embedding_task_id)
            print(f"   🔗 Queued user message embedding: {user_embedding_task_id}")

        # Queue AI response embedding
        if ai_content:
            # Ensure ai_content is a string for embedding
            ai_content_text = ai_content
            if isinstance(ai_content, dict):
                import json

                ai_content_text = json.dumps(ai_content, indent=2)
            elif not isinstance(ai_content, str):
                ai_content_text = str(ai_content)

            ai_embedding_task_id = await real_time_context_service.queue_context_update(
                user_id=user_id,
                interaction_id=interaction_id,
                conversation_id=str(ai_conv_id),
                update_type="embedding",
                payload={
                    "conversation_id": str(ai_conv_id),
                    "text": ai_content_text,
                    "title": f"AI response in {interaction_id}",
                    "metadata": enhanced_metadata,
                },
                priority=2,
            )
            embedding_tasks.append(ai_embedding_task_id)
            print(f"   🔗 Queued AI response embedding: {ai_embedding_task_id}")

        # === STEP 3: Queue conversation context update ===
        context_task_id = await real_time_context_service.queue_context_update(
            user_id=user_id,
            interaction_id=interaction_id,
            conversation_id=str(ai_conv_id),
            update_type="conversation_context",
            payload={
                "context_data": {
                    "user_message": message,
                    "ai_response": ai_content,
                    "relevance_score": 0.9,
                    "importance_score": 0.8,
                    "context_type": "conversation_pair",
                }
            },
            priority=3,
        )
        print(f"   💾 Queued conversation context update: {context_task_id}")

        # === STEP 4: Monitor completion (optional) ===
        # Wait a bit for critical updates to complete
        await asyncio.sleep(2)

        # Check status of critical updates
        semantic_status = await real_time_context_service.get_update_status(
            semantic_task_id
        )
        if semantic_status:
            print(
                f"   📊 Semantic summary status: {semantic_status.get('status', 'unknown')}"
            )

        print(
            f"✅ Enhanced background operations queued successfully for interaction {interaction_id}"
        )
        print(
            f"   Total tasks queued: {1 + len(embedding_tasks) + 1}"
        )  # semantic + embeddings + context

    except Exception as e:
        print(
            f"❌ Enhanced background operations error for interaction {interaction_id}: {e}"
        )
        import traceback

        traceback.print_exc()

        # Fallback to original method if enhanced fails
        print(f"🔄 Falling back to original background operations...")
        await _background_operations(
            user_conv_id, ai_conv_id, user_id, interaction_id, message, ai_content
        )


async def _background_operations(
    user_conv_id: int,
    ai_conv_id: int,
    user_id: str,
    interaction_id: str,
    message: str,
    ai_content: str,
):
    """Original background operations for conversation processing (fallback)"""
    try:
        print(
            f"🔄 Starting fallback background operations for interaction {interaction_id}"
        )

        # === STEP 1: Create conversation-level summary ===
        try:
            print(f"   📝 Creating conversation summary...")
            conversation_summary = await langchain_service.summarize_conversation(
                user_message=message, ai_response=ai_content
            )
            print(f"   ✅ Conversation summary created")
            print(f"      Topics: {conversation_summary.get('main_topics', [])}")
            print(f"      Facts: {len(conversation_summary.get('key_facts', []))}")
        except Exception as sum_error:
            print(f"   ⚠️  Semantic summary creation failed: {sum_error}")

        # === STEP 2: Update interaction-level running summary ===
        try:
            print(f"   🔄 Updating interaction-level summary...")
            async with AsyncSessionLocal() as db:
                # Get current interaction
                result = await db.execute(
                    select(Interaction).where(Interaction.id == str(interaction_id))
                )
                interaction = result.scalar_one_or_none()

                if interaction:
                    current_summary = interaction.semantic_summary

                    # Update the running summary
                    updated_summary = (
                        await langchain_service.update_interaction_summary(
                            current_summary=current_summary,
                            new_user_message=message,
                            new_ai_response=ai_content,
                        )
                    )

                    # Save updated summary to database
                    interaction.semantic_summary = updated_summary
                    await db.commit()

                    print(f"   ✅ Interaction summary updated and saved")
                    print(f"      Topics: {len(updated_summary.get('key_topics', []))}")
                    print(
                        f"      Facts: {len(updated_summary.get('accumulated_facts', []))}"
                    )
        except Exception as sum_update_error:
            print(f"   ⚠️  Interaction summary update failed: {sum_update_error}")

        # === STEP 3: Create embeddings with enhanced metadata ===
        embedding_tasks = []

        # Prepare metadata with summary information
        enhanced_metadata = {
            "interaction_id": str(interaction_id),
        }

        if conversation_summary:
            enhanced_metadata["summary"] = conversation_summary.get(
                "semantic_summary", ""
            )
            enhanced_metadata["topics"] = conversation_summary.get("main_topics", [])
            enhanced_metadata["facts"] = conversation_summary.get("key_facts", [])

        # Create embeddings for user message
        if message:
            embedding_tasks.append(
                langchain_service.upsert_embedding(
                    conv_id=str(user_conv_id),
                    user_id=user_id,
                    text=message,
                    title=f"User message in {interaction_id}",
                    metadata=enhanced_metadata,
                )
            )

        # Create embeddings for AI response
        if ai_content:
            # Ensure ai_content is a string for embedding
            ai_content_text = ai_content
            if isinstance(ai_content, dict):
                import json

                ai_content_text = json.dumps(ai_content, indent=2)
            elif not isinstance(ai_content, str):
                ai_content_text = str(ai_content)

            embedding_tasks.append(
                langchain_service.upsert_embedding(
                    conv_id=str(ai_conv_id),
                    user_id=user_id,
                    text=ai_content_text,
                    title=f"AI response in {interaction_id}",
                    metadata=enhanced_metadata,
                )
            )

        # Execute embedding tasks in parallel
        if embedding_tasks:
            print(f"   🔗 Creating {len(embedding_tasks)} embeddings...")
            embedding_results = await asyncio.gather(
                *embedding_tasks, return_exceptions=True
            )

            successful_embeddings = 0
            for i, result in enumerate(embedding_results):
                if isinstance(result, Exception):
                    print(f"   ⚠️  Embedding {i+1} failed: {result}")
                elif result:
                    successful_embeddings += 1

            print(
                f"   ✅ Created {successful_embeddings}/{len(embedding_tasks)} embeddings successfully"
            )

        print(
            f"✅ Fallback background operations completed successfully for interaction {interaction_id}"
        )

    except Exception as e:
        print(
            f"❌ Fallback background operations error for interaction {interaction_id}: {e}"
        )
        import traceback

        traceback.print_exc()
        pass


async def cancel_ai_generation(
    user_id: str, interaction_id: str, db: AsyncSession
) -> Dict[str, Any]:
    """Cancel ongoing AI generation for a user's interaction"""
    print(f"\n{'='*60}")
    print(f"🛑 CANCELLATION REQUESTED")
    print(f"{'='*60}")
    print(f"👤 User ID: {user_id}")
    print(f"💬 Interaction ID: {interaction_id}")
    print(f"{'='*60}\n")

    try:
        # Check if the interaction exists and belongs to the user
        result = await db.execute(
            select(Interaction).where(
                Interaction.id == interaction_id, Interaction.user_id == user_id
            )
        )
        interaction = result.scalar_one_or_none()

        if not interaction:
            return {
                "success": False,
                "message": "Interaction not found or access denied",
            }

        # Check if there's an active task for this interaction
        task_key = f"{user_id}_{interaction_id}"
        if task_key in active_generation_tasks:
            task = active_generation_tasks[task_key]
            if not task.done():
                task.cancel()
                print(f"✅ Cancelled active AI generation task for {task_key}")
            del active_generation_tasks[task_key]

        # Find the most recent processing conversation for this interaction
        conv_result = await db.execute(
            select(Conversation)
            .where(
                Conversation.interaction_id == interaction_id,
                Conversation.role == ConversationRole.AI,
                Conversation.status == "processing",
            )
            .order_by(Conversation.created_at.desc())
            .limit(1)
        )
        processing_conv = conv_result.scalar_one_or_none()

        if processing_conv:
            # Mark as cancelled
            processing_conv.status = "cancelled"
            processing_conv.error_message = "Generation stopped by user"
            processing_conv.content = {
                "type": "text",
                "result": {"content": "Response generation was stopped by user."},
            }
            await db.commit()
            print(f"✅ Marked conversation {processing_conv.id} as cancelled")

            # Send notification that generation was cancelled
            try:
                from app.api.sse_routes import notify_ai_response_ready_sse

                await notify_ai_response_ready_sse(
                    user_id=user_id,
                    interaction_id=interaction_id,
                    ai_response="Response generation was stopped by user.",
                )
                print(f"✅ Sent cancellation notification via SSE")
            except Exception as e:
                print(f"⚠️  Failed to send cancellation notification: {e}")

            return {
                "success": True,
                "message": "Generation cancelled successfully",
                "interaction_id": interaction_id,
            }
        else:
            print(f"⚠️  No processing conversation found to cancel")
            return {
                "success": False,
                "message": "No active generation found",
                "interaction_id": interaction_id,
            }

    except Exception as e:
        print(f"❌ Error cancelling AI generation: {e}")
        await db.rollback()
        return {"success": False, "message": f"Failed to cancel generation: {str(e)}"}


async def _send_notifications_background(
    user_id: str, interaction_id: str, conversation_id: str
):
    """Send notifications in background without blocking main flow"""
    print(f"\n{'='*60}")
    print(f"📤 SENDING MESSAGE_RECEIVED NOTIFICATION")
    print(f"{'='*60}")
    print(f"👤 User ID: {user_id}")
    print(f"💬 Interaction ID: {interaction_id}")
    print(f"🆔 Conversation ID: {conversation_id}")
    print(f"{'='*60}\n")

    # Try WebSocket first
    try:
        from app.api.websocket_routes import notify_message_received

        await notify_message_received(
            user_id=user_id,
            interaction_id=interaction_id,
            conversation_id=conversation_id,
        )
        print(f"✅ WebSocket notification sent successfully for user {user_id}")
    except Exception as ws_error:
        print(f"⚠️  WebSocket notification failed: {ws_error}")
        # Fallback to SSE
        try:
            from app.api.sse_routes import notify_message_received_sse

            await notify_message_received_sse(
                user_id=user_id,
                interaction_id=interaction_id,
                conversation_id=conversation_id,
            )
            print(f"✅ SSE notification sent successfully for user {user_id}")
        except Exception as sse_error:
            print(f"❌ SSE notification failed: {sse_error}")
            import traceback

            traceback.print_exc()
            print(f"❌ All notification methods failed for user {user_id}")


async def _send_ai_response_notification(
    user_id: str, interaction_id: str, ai_response: str
):
    """Send AI response notification in background without blocking main flow"""
    print(f"\n{'='*60}")
    print(f"📤 SENDING AI_RESPONSE_READY NOTIFICATION")
    print(f"{'='*60}")
    print(f"👤 User ID: {user_id}")
    print(f"💬 Interaction ID: {interaction_id}")
    # Handle both string and dict responses
    if isinstance(ai_response, dict):
        response_preview = str(ai_response)[:100] + "..."
        response_length = len(str(ai_response))
    else:
        response_preview = (
            ai_response[:100] + "..." if len(ai_response) > 100 else ai_response
        )
        response_length = len(ai_response)

    print(f"📝 Response Length: {response_length} characters")
    print(f"📄 Response Preview: {response_preview}")
    print(f"{'='*60}\n")

    # Try WebSocket first
    try:
        from app.api.websocket_routes import notify_ai_response_ready

        await notify_ai_response_ready(
            user_id=user_id,
            interaction_id=interaction_id,
            ai_response=ai_response,
        )
        print(
            f"✅ WebSocket AI response notification sent successfully for user {user_id}"
        )
    except Exception as ws_error:
        print(f"⚠️  WebSocket AI response notification failed: {ws_error}")
        # Fallback to SSE
        try:
            from app.api.sse_routes import notify_ai_response_ready_sse

            await notify_ai_response_ready_sse(
                user_id=user_id,
                interaction_id=interaction_id,
                ai_response=ai_response,
            )
            print(
                f"✅ SSE AI response notification sent successfully for user {user_id}"
            )
        except Exception as sse_error:
            print(f"❌ SSE AI response notification failed: {sse_error}")
            import traceback

            traceback.print_exc()
            print(f"❌ All AI response notification methods failed for user {user_id}")


async def _extract_interaction_metadata_fast(
    interaction: Interaction, content_text, original_message: str = ""
):
    """Fast metadata extraction with dedicated title generation using langchain_service"""
    try:
        # Since title and summary_title are no longer in the prompt, always use AI generation
        # Check if we need to update the title
        needs_title_update = (
            not interaction.title
            or not interaction.summary_title
            or interaction.title in ["Study Session", "New Interaction"]
            or len(interaction.title) < 5
        )

        if needs_title_update:
            print(
                f"🔍 Title update needed. Current title: '{interaction.title}', summary_title: '{interaction.summary_title}'"
            )

            # Handle different content types for response preview
            if isinstance(content_text, dict):
                # For dictionary responses (like MCQ), convert to string for preview
                response_preview = str(content_text)[:300]
            elif isinstance(content_text, str):
                response_preview = content_text[:300]
            else:
                response_preview = str(content_text)[:300]

            print(
                f"🔍 Calling title generation with message: '{original_message}', response_preview: '{response_preview}'"
            )

            # Use AI generation for title and summary
            try:
                title, summary_title = (
                    await langchain_service.generate_interaction_title(
                        message=original_message,
                        response_preview=response_preview,
                    )
                )
                print(
                    f"🔍 Generated title: '{title}', summary_title: '{summary_title}'"
                )

            except Exception as ai_title_error:
                print(f"⚠️ AI title generation failed: {ai_title_error}")
                # Fallback to simple title generation
                if original_message:
                    title = original_message[:40].strip()
                    summary_title = f"Help with {title.lower()}"
                else:
                    title = "Study Session"
                    summary_title = "Educational assistance"

            # Update interaction with generated titles
            if title:
                interaction.title = title
                print(f"✅ Set interaction title: '{title}'")
            else:
                print(f"⚠️ No title generated")

            if summary_title:
                interaction.summary_title = summary_title
                print(f"✅ Set interaction summary_title: '{summary_title}'")
            else:
                print(f"⚠️ No summary title generated")
        else:
            print(f"✅ Title and summary already exist, skipping generation")

    except Exception as e:
        print(f"Metadata extraction error: {e}")
        # Final fallback: create basic title from message
        if original_message and not interaction.title:
            interaction.title = original_message[:40].strip()


def _format_mcq_response(mcq_data: dict) -> str:
    """Format MCQ response into readable text for frontend display"""
    try:
        print(f"🔧 _format_mcq_response called with: {mcq_data}")
        title = mcq_data.get("title", "MCQ Questions")
        questions = mcq_data.get("_result", {}).get("questions", [])
        print(f"🔧 Extracted title: {title}")
        print(f"🔧 Extracted questions: {questions}")

        if not questions:
            result = f"**{title}**\n\nNo questions found in the document."
            print(f"🔧 No questions found, returning: {result}")
            return result

        formatted_response = f"**{title}**\n\n"

        for i, question_data in enumerate(questions, 1):
            question_text = question_data.get("question", "")
            options = question_data.get("options", {})
            answer = question_data.get("answer", "")
            explanation = question_data.get("explanation", "")

            formatted_response += f"**{i}. {question_text}**\n"

            if options:
                for option_key, option_value in options.items():
                    formatted_response += f"   {option_key.upper()}. {option_value}\n"

            if answer:
                formatted_response += f"\n**Answer: {answer.upper()}**\n"

            if explanation:
                formatted_response += f"**Explanation:** {explanation}\n"

            formatted_response += "\n---\n\n"

        result = formatted_response.strip()

        # Safety check: ensure we never return empty content
        if not result or result.strip() == "":
            result = f"**MCQ Questions**\n\nQuestions were generated but could not be formatted properly."

        print(f"🔧 _format_mcq_response returning: {result}")
        return result

    except Exception as e:
        print(f"❌ Error formatting MCQ response: {e}")
        result = f"**MCQ Questions**\n\nError formatting questions: {str(e)}"
        print(f"🔧 Error case returning: {result}")
        return result


def _process_ai_content_fast(content_text) -> tuple[str, str]:
    """Fast AI content processing with enhanced formatting and escaped character handling"""
    print(
        f"🔧 PROCESS AI CONTENT START: type={type(content_text)}, length={len(str(content_text)) if content_text else 0}"
    )

    # Initialize with safe defaults
    ai_content_type = "written"
    ai_result_content = str(content_text) if content_text else "No content provided"

    # Ensure we always return valid content
    if not content_text:
        print(f"🔧 No content provided, returning default")
        return ai_content_type, ai_result_content

    # Handle structured responses (dictionaries) from document analysis
    if isinstance(content_text, dict):
        print(f"🔧 PROCESSING DICT: {content_text}")
        # This is a structured response from document analysis
        ai_content_type = (
            "mcq" if content_text.get("type") == "mcq" else "document_analysis"
        )

        # Convert structured response to readable format for frontend
        if content_text.get("type") == "mcq":
            print(f"🔧 CALLING _format_mcq_response with: {content_text}")
            ai_result_content = _format_mcq_response(content_text)
            print(f"🔧 _format_mcq_response returned: {ai_result_content}")
        else:
            # For other structured responses, convert to JSON
            import json

            ai_result_content = json.dumps(content_text, indent=2)

        print(
            f"🔧 DICT RESULT: type={ai_content_type}, length={len(ai_result_content) if ai_result_content else 0}"
        )
        print(f"🔧 FINAL RESULT CONTENT: {ai_result_content}")
        return ai_content_type, ai_result_content

    # Handle string responses (regular conversations)
    if content_text and isinstance(content_text, str):
        print(f"🔧 PROCESSING STRING: {content_text[:100]}...")
        # Remove escaped characters that show as backslashes
        ai_result_content = content_text.replace("\\(", "(").replace("\\)", ")")
        ai_result_content = ai_result_content.replace("\\[", "[").replace("\\]", "]")
        ai_result_content = ai_result_content.replace("\\\\", "")
        ai_result_content = ai_result_content.replace("\\ ", " ")

        # Clean up common LaTeX patterns
        ai_result_content = ai_result_content.replace("\\geq", "≥")
        ai_result_content = ai_result_content.replace("\\leq", "≤")
        ai_result_content = ai_result_content.replace("\\times", "×")
        ai_result_content = ai_result_content.replace("\\div", "÷")

    # Detect MCQ patterns in plain text content (only for strings)
    if (
        content_text
        and isinstance(content_text, str)
        and not content_text.strip().startswith("{")
    ):
        # Check for MCQ patterns: numbered questions with Answer: sections
        import re

        # Pattern 1: Numbered questions with "Answer:" pattern
        answer_pattern = r"\d+\.\s+.+?Answer:\s*[A-Za-z]"
        if re.search(answer_pattern, content_text, re.MULTILINE | re.DOTALL):
            ai_content_type = "mcq"
            print(f"🎯 MCQ DETECTED: Answer pattern found")
        else:
            # Pattern 2: Multiple numbered questions with options A, B, C, D
            option_pattern = (
                r"\d+\.\s+.+?[A-D]\.\s+.+?[A-D]\.\s+.+?[A-D]\.\s+.+?[A-D]\."
            )
            if re.search(option_pattern, content_text, re.MULTILINE | re.DOTALL):
                ai_content_type = "mcq"
                print(f"🎯 MCQ DETECTED: Multiple choice options found")
            else:
                # Pattern 3: Explanation sections (common in MCQ)
                explanation_pattern = r"Explanation:\s+.+"
                if re.search(explanation_pattern, content_text, re.MULTILINE):
                    ai_content_type = "mcq"
                    print(f"🎯 MCQ DETECTED: Explanation sections found")

        print(f"🔍 MCQ Detection result: {ai_content_type}")

    try:
        if content_text and content_text.strip().startswith("{"):
            parsed_response = json.loads(content_text)
            if isinstance(parsed_response, dict):
                ai_content_type = parsed_response.get("type", "written")
                result_content = parsed_response.get("_result", {})

                if isinstance(result_content, dict):
                    if "content" in result_content:
                        ai_result_content = result_content["content"]
                    elif "questions" in result_content:
                        # Enhanced MCQ formatting - show ALL questions
                        questions = result_content[
                            "questions"
                        ]  # Remove limit to show all questions
                        formatted_questions = []

                        for i, q in enumerate(questions, 1):
                            question_text = q.get("question", "")
                            options = q.get("options", {})
                            answer = q.get("answer", "")
                            explanation = q.get("explanation", "")

                            # Format question with better structure
                            formatted_q = f"{i}. {question_text}\n\n"

                            # Format options if they exist
                            if options and isinstance(options, dict):
                                for opt_key, opt_value in list(options.items())[:4]:
                                    formatted_q += (
                                        f"{opt_key.upper()}. {str(opt_value)}\n"
                                    )
                                formatted_q += "\n"

                            # Add answer with clear formatting
                            if answer:
                                formatted_q += f"Answer: {answer}\n\n"

                            # Add explanation with clear formatting
                            if explanation:
                                formatted_q += f"Explanation: {explanation}\n"

                            formatted_questions.append(formatted_q)

                        ai_result_content = "\n".join(formatted_questions)

        # Apply character cleaning to any processed content
        if ai_result_content:
            ai_result_content = ai_result_content.replace("\\(", "(").replace(
                "\\)", ")"
            )
            ai_result_content = ai_result_content.replace("\\[", "[").replace(
                "\\]", "]"
            )
            ai_result_content = ai_result_content.replace("\\\\", "")
            ai_result_content = ai_result_content.replace("\\ ", " ")

    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # FINAL SAFETY CHECK: Ensure we always return valid content
    if not ai_result_content or (
        isinstance(ai_result_content, str) and ai_result_content.strip() == ""
    ):
        print(f"🔧 FINAL SAFETY: No valid content, using fallback")
        ai_content_type = "error"
        ai_result_content = (
            str(content_text) if content_text else "AI response could not be processed"
        )

    print(
        f"🔧 PROCESS AI CONTENT END: type={ai_content_type}, length={len(str(ai_result_content)) if ai_result_content else 0}"
    )
    return ai_content_type, ai_result_content
