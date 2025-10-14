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
from datetime import datetime
from app.models.user import User
from app.models.interaction import Interaction, Conversation, ConversationRole
from app.models.media import Media
from app.models.subscription import PointTransaction
from app.services.langchain_service import langchain_service
from app.services.cache_service import (
    cache_user_context,
    get_cached_user_context,
    cache_interaction_data,
    get_cached_interaction_data,
)
from app.core.database import AsyncSessionLocal
from app.core.config import settings
import asyncio

# Global task tracking for AI generation cancellation
active_generation_tasks: Dict[str, asyncio.Task] = {}


async def process_conversation_message(
    *,
    user: User,
    interaction: Optional[Interaction],
    message: Optional[str],
    media_files: Optional[List[Dict[str, str]]],
    max_tokens: int = 800,  # Reduced from 1000
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

    # Use a single DB session for the entire operation
    try:
        async with AsyncSessionLocal() as db:
            user_id = user.id
            interaction_id = interaction.id if interaction else None

            # === PHASE 1: PARALLEL SETUP & MEDIA PROCESSING ===
            async def process_media_parallel():
                """Process media files in parallel"""
                if not media_files:
                    return [], []

                media_objects = []
                media_urls = []

                # Process all media files concurrently
                media_tasks = []
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
                        # Queue database lookup
                        media_tasks.append(
                            db.execute(select(Media).where(Media.id == media_id))
                        )

                # Execute all DB queries in parallel
                if media_tasks:
                    media_results = await asyncio.gather(
                        *media_tasks, return_exceptions=True
                    )
                    for result in media_results:
                        if not isinstance(result, Exception):
                            media = result.scalar_one_or_none()
                            if media:
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
            async def run_guardrails():
                """Run guardrail checks"""
                if not (message or media_urls):
                    return None
                try:
                    return await langchain_service.check_guardrails(
                        message or "", media_urls
                    )
                except Exception:
                    return None

            async def get_context_fast():
                """Enhanced context retrieval using multi-level strategy"""
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
                    print(f"Enhanced context retrieval error: {e}")
                    return ""

            # Run guardrails and context retrieval in parallel
            guardrail_result, context_text = await asyncio.gather(
                run_guardrails(), get_context_fast(), return_exceptions=True
            )

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
            if media_urls and len(media_urls) > 0:
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

                for i, media_url in enumerate(media_urls):
                    print(
                        f"🔍 Processing document {i+1}/{len(media_urls)}: {media_url}"
                    )
                    print(f"📊 Using {attachment_tokens} tokens for this document")

                    try:
                        # Get media object for this URL
                        media_obj = media_objects[i] if i < len(media_objects) else None
                        media_id = media_obj.id if media_obj else f"media_{i}"

                        # Use comprehensive document processing
                        document_analysis = await document_integration_service.process_document_comprehensive(
                            media_id=media_id,
                            interaction_id=str(interaction.id),
                            user_id=str(user_id),
                            file_url=media_url,
                            max_tokens=attachment_tokens,
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
                            "token": attachment_tokens,  # Estimated tokens
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

                        analysis_results.append(analysis_result)
                        processed_documents.append(document_analysis)
                        total_tokens += attachment_tokens

                        print(f"✅ Document {i+1} processed successfully")
                        print(f"   Type: {document_analysis.document_type}")
                        print(f"   Questions: {document_analysis.total_questions}")
                        print(f"   Topics: {document_analysis.main_topics}")
                        print(f"   Chunks: {len(document_analysis.chunks)}")

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
                                analysis_results.append(basic_analysis)
                                total_tokens += basic_analysis.get("token", 0)
                        except Exception as basic_error:
                            print(f"❌ Basic analysis also failed: {basic_error}")
                            continue

                if analysis_results:
                    print(
                        f"✅ Document processing completed: {len(analysis_results)} results"
                    )
                    print(f"📊 Total tokens used: {total_tokens}")
                    print(f"📄 Processed documents: {len(processed_documents)}")

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
            print(f"💬 Generating conversation response...")

            # Create a task for AI generation that can be cancelled
            task_key = f"{user_id}_{interaction.id}"

            async def generate_ai_response():
                return await langchain_service.generate_conversation_response(
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

            # Create and track the task
            generation_task = asyncio.create_task(generate_ai_response())
            active_generation_tasks[task_key] = generation_task

            try:
                ai_response, input_tokens, output_tokens, total_tokens = (
                    await generation_task
                )

                print(f"✅ AI response generated successfully")
                print(f"   Input tokens: {input_tokens}")
                print(f"   Output tokens: {output_tokens}")
                print(f"   Total tokens: {total_tokens}")

            except asyncio.CancelledError:
                print(
                    f"🛑 AI generation was cancelled for interaction {interaction.id}"
                )
                # Clean up the task from tracking
                if task_key in active_generation_tasks:
                    del active_generation_tasks[task_key]
                return {
                    "success": False,
                    "message": "AI generation was cancelled",
                    "interaction_id": str(interaction.id),
                }
            except Exception as e:
                print(f"❌ AI response generation failed: {e}")
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

            # Create conversation entries
            print(f"💾 Creating conversation entries...")

            # Create user conversation entry
            user_conv = Conversation(
                interaction_id=str(interaction.id),
                role=ConversationRole.USER,
                content={
                    "type": "text",
                    "_result": {
                        "note": message or "User uploaded document",
                        "media_urls": media_urls or [],
                    },
                },
                status="completed",
            )

            db.add(user_conv)
            await db.flush()

            # Create AI conversation entry
            ai_conv = Conversation(
                interaction_id=str(interaction.id),
                role=ConversationRole.AI,
                content={"type": "text", "_result": {"note": ai_response}},
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tokens_used=total_tokens,
                points_cost=langchain_service.calculate_points_cost(total_tokens),
                status="completed",
            )

            db.add(ai_conv)
            await db.flush()

            # Update interaction title if needed
            if not interaction.title:
                try:
                    title, summary_title = (
                        await langchain_service.generate_interaction_title(
                            message or "", ai_response[:200]
                        )
                    )
                    if title:
                        interaction.title = title
                    if summary_title:
                        interaction.summary_title = summary_title
                    await db.commit()
                    print(f"✅ Updated interaction title: {title}")
                except Exception as title_error:
                    pass
                    # Don't let title generation failure break the main AI response

            # Commit all changes
            await db.commit()

            print(f"✅ Conversation entries created successfully")
            print(f"   User conversation ID: {user_conv.id}")
            print(f"   AI conversation ID: {ai_conv.id}")

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

            return {
                "success": True,
                "message": "Response generated successfully",
                "conversation_id": str(ai_conv.id),
                "ai_response": ai_response,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "points_cost": ai_conv.points_cost,
            }

    except Exception as e:
        print(f"❌ Error in process_conversation_message: {e}")
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
            ai_embedding_task_id = await real_time_context_service.queue_context_update(
                user_id=user_id,
                interaction_id=interaction_id,
                conversation_id=str(ai_conv_id),
                update_type="embedding",
                payload={
                    "conversation_id": str(ai_conv_id),
                    "text": ai_content,
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
            embedding_tasks.append(
                langchain_service.upsert_embedding(
                    conv_id=str(ai_conv_id),
                    user_id=user_id,
                    text=ai_content,
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
    """Send background notifications (placeholder)"""
    try:
        # This would integrate with your notification system
        print(
            f"📱 Sending notification to user {user_id} for conversation {conversation_id}"
        )
        # Add your notification logic here
        pass
    except Exception as e:
        print(f"❌ Notification failed: {e}")
