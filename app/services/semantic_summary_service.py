"""
DEPRECATED: Semantic Summary Service

This service was used for creating and updating semantic summaries of conversations.
It has been deprecated in the RAG streamlining - vector search now handles all
context retrieval.

All functions now return minimal fallback responses and do NOT call LLMs.
"""

import json
import re
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from sqlalchemy import select, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.models.interaction import Interaction


class SemanticSummaryService:
    """
    DEPRECATED: This service is no longer used in the streamlined RAG system.
    All functions return minimal fallback responses and do NOT call LLMs.
    Vector search now handles all context retrieval.
    """

    def __init__(self):
        self.max_summary_length = 500
        self.max_facts = 10
        self.max_topics = 8

    async def create_conversation_summary(
        self, user_message: str, ai_response: str
    ) -> Dict[str, Any]:
        """
        DEPRECATED: Returns minimal fallback response without calling LLM.
        Semantic summaries were removed in RAG streamlining.
        """
        print(
            "⚠️ create_conversation_summary is DEPRECATED (RAG streamlining) - no LLM call"
        )
        return {
            "key_facts": [],
            "main_topics": [],
            "semantic_summary": "",
            "important_terms": [],
            "context_for_future": "",
            "question_numbers": [],
            "learning_progress": "",
            "potential_follow_ups": [],
            "difficulty_level": "unknown",
            "subject_area": "unknown",
        }

    async def _create_conversation_summary_legacy(
        self, user_message: str, ai_response: str
    ) -> Dict[str, Any]:
        """
        LEGACY: Original implementation kept for reference only.
        This function is NOT called - see create_conversation_summary above.
        """
        try:
            # Use the enhanced conversation summarization chain
            from app.config.langchain_config import StudyGuruConfig

            chain = StudyGuruConfig.CHAINS.get_conversation_summarization_chain()

            # Truncate inputs to prevent token overflow
            user_msg_truncated = (
                user_message[:500] if len(user_message) > 500 else user_message
            )
            # Handle both string and dict responses for truncation
            if isinstance(ai_response, dict):
                ai_resp_truncated = str(ai_response)[:1000]
            else:
                ai_resp_truncated = (
                    ai_response[:1000] if len(ai_response) > 1000 else ai_response
                )

            # Try multiple attempts with progressive fallback
            result = None
            for attempt in range(2):  # Try twice with different token limits
                try:
                    # Adjust token limits based on attempt
                    if attempt == 0:
                        # First attempt: use full chain with increased tokens
                        result = await chain.ainvoke(
                            {
                                "user_message": user_msg_truncated,
                                "ai_response": ai_resp_truncated,
                            }
                        )
                    else:
                        # Second attempt: use even more truncated inputs
                        user_msg_ultra_truncated = user_msg_truncated[:200]
                        ai_resp_ultra_truncated = ai_resp_truncated[:400]
                        result = await chain.ainvoke(
                            {
                                "user_message": user_msg_ultra_truncated,
                                "ai_response": ai_resp_ultra_truncated,
                            }
                        )

                    # Check if result is valid
                    if (
                        result
                        and isinstance(result, dict)
                        and result.get("semantic_summary")
                    ):
                        break
                    else:
                        print(
                            f"⚠️ Attempt {attempt + 1}: Invalid result format: {result}"
                        )

                except Exception as chain_error:
                    print(f"⚠️ Attempt {attempt + 1}: Chain error: {chain_error}")
                    # Check if it's a length limit error
                    if "length limit" in str(
                        chain_error
                    ).lower() or "completion_tokens" in str(chain_error):
                        print(
                            "⚠️ Length limit error detected, trying with shorter inputs"
                        )
                        continue
                    # Check if it's a JSON parsing error
                    elif (
                        "json" in str(chain_error).lower()
                        or "parse" in str(chain_error).lower()
                    ):
                        print("⚠️ JSON parsing error detected, using fallback summary")
                        break

            # Validate and enhance the result
            if result and isinstance(result, dict):
                validated_result = self._validate_and_enhance_summary(result)
                return validated_result
            else:
                print("⚠️ All attempts failed, using fallback summary")
                return self._get_fallback_summary(user_message, ai_response)

        except Exception as e:
            print(f"Error creating conversation summary: {e}")
            # Only print traceback for non-length-limit errors to reduce noise
            if "length limit" not in str(e).lower() and "completion_tokens" not in str(
                e
            ):
                import traceback

                traceback.print_exc()

            return self._get_fallback_summary(user_message, ai_response)

    async def update_interaction_summary(
        self,
        current_summary: Optional[Dict[str, Any]],
        new_user_message: str,
        new_ai_response: str,
    ) -> Dict[str, Any]:
        """
        DEPRECATED: Returns minimal fallback response without calling LLM.
        Semantic summaries were removed in RAG streamlining.
        """
        print(
            "⚠️ update_interaction_summary is DEPRECATED (RAG streamlining) - no LLM call"
        )
        return current_summary or {
            "updated_summary": "",
            "key_topics": [],
            "recent_focus": "",
            "accumulated_facts": [],
            "question_numbers": [],
            "learning_progression": "",
            "difficulty_trend": "unknown",
            "learning_patterns": [],
            "struggling_areas": [],
            "mastered_concepts": [],
            "version": 1,
            "last_updated": datetime.now().isoformat(),
        }

    def _validate_and_enhance_summary(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and enhance a semantic summary
        """
        try:
            # Ensure required fields exist
            validated = {
                "updated_summary": summary.get("updated_summary", ""),
                "key_topics": summary.get("key_topics", []),
                "recent_focus": summary.get("recent_focus", ""),
                "accumulated_facts": summary.get("accumulated_facts", []),
                "question_numbers": summary.get("question_numbers", []),
                "learning_progression": summary.get("learning_progression", ""),
                "difficulty_trend": summary.get("difficulty_trend", "beginner"),
                "learning_patterns": summary.get("learning_patterns", []),
                "struggling_areas": summary.get("struggling_areas", []),
                "mastered_concepts": summary.get("mastered_concepts", []),
                "version": summary.get("version", 1),
                "last_updated": summary.get("last_updated", datetime.now().isoformat()),
            }

            # Check if we have a semantic_summary from the result and use it FIRST
            if "semantic_summary" in summary and summary["semantic_summary"]:
                validated["updated_summary"] = summary["semantic_summary"]
                print(
                    f"✅ Using semantic_summary from result: {validated['updated_summary'][:100]}..."
                )
                print(
                    f"🔍 Summary length: {len(validated['updated_summary'])} characters"
                )

            # Validate and clean data
            validated["updated_summary"] = self._clean_text(
                validated["updated_summary"]
            )
            validated["recent_focus"] = self._clean_text(validated["recent_focus"])
            validated["learning_progression"] = self._clean_text(
                validated["learning_progression"]
            )

            # Ensure summary is not empty or too short
            summary_text = validated["updated_summary"]
            if not summary_text or len(summary_text) < 10:
                print(f"⚠️ Semantic summary too short or empty, generating fallback")
                print(
                    f"🔍 Summary text: '{summary_text}', length: {len(summary_text) if summary_text else 0}"
                )
                validated["updated_summary"] = (
                    "Educational conversation covering various topics and concepts."
                )
                validated["key_topics"] = validated["key_topics"] or [
                    "General Discussion"
                ]
                validated["accumulated_facts"] = validated["accumulated_facts"] or [
                    "Learning in progress"
                ]
            else:
                print(
                    f"✅ Semantic summary validation passed: {len(summary_text)} characters"
                )

            # Ensure lists are properly formatted
            for list_field in [
                "key_topics",
                "accumulated_facts",
                "question_numbers",
                "learning_patterns",
                "struggling_areas",
                "mastered_concepts",
            ]:
                if not isinstance(validated[list_field], list):
                    validated[list_field] = []
                else:
                    # Clean and limit list items
                    validated[list_field] = [
                        self._clean_text(str(item)) for item in validated[list_field]
                    ][
                        : (
                            self.max_facts
                            if list_field == "accumulated_facts"
                            else self.max_topics
                        )
                    ]

            # Validate difficulty trend
            if validated["difficulty_trend"] not in [
                "beginner",
                "intermediate",
                "advanced",
            ]:
                validated["difficulty_trend"] = "beginner"

            # Extract question numbers from text if not provided
            if not validated["question_numbers"]:
                validated["question_numbers"] = self._extract_question_numbers(
                    validated["updated_summary"] + " " + validated["recent_focus"]
                )

            # Compress summary if too long
            if len(validated["updated_summary"]) > self.max_summary_length:
                validated["updated_summary"] = self._compress_summary(
                    validated["updated_summary"]
                )

            return validated

        except Exception as e:
            print(f"Error validating summary: {e}")
            return self._get_fallback_summary("", "")

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Remove any non-printable characters
        text = "".join(char for char in text if char.isprintable() or char.isspace())

        return text

    def _extract_question_numbers(self, text: str) -> List[int]:
        """Extract question numbers from text"""
        question_numbers = []
        patterns = [
            r"question\s+(\d+)",
            r"problem\s+(\d+)",
            r"mcq\s+(\d+)",
            r"equation\s+(\d+)",
            r"(\d+)\.\s*$",  # Number at end of line
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    num = int(match)
                    if num not in question_numbers:
                        question_numbers.append(num)
                except ValueError:
                    continue

        return sorted(question_numbers)

    def _compress_summary(self, summary: str) -> str:
        """Compress summary to fit within length limit"""
        if len(summary) <= self.max_summary_length:
            return summary

        # Try to find a good breaking point
        sentences = summary.split(". ")
        compressed = ""

        for sentence in sentences:
            if len(compressed + sentence + ". ") <= self.max_summary_length:
                compressed += sentence + ". "
            else:
                break

        return compressed.strip()

    def _get_fallback_summary(
        self, user_message: str, ai_response: str
    ) -> Dict[str, Any]:
        """Get a fallback summary when processing fails"""
        # Try to extract some basic information from the inputs
        topics = []
        facts = []

        # Extract basic topics from user message
        if user_message:
            user_lower = user_message.lower()
            if any(
                word in user_lower
                for word in ["math", "mathematics", "equation", "solve"]
            ):
                topics.append("Mathematics")
            if any(
                word in user_lower
                for word in ["science", "physics", "chemistry", "biology"]
            ):
                topics.append("Science")
            if any(
                word in user_lower
                for word in ["english", "language", "grammar", "writing"]
            ):
                topics.append("Language")
            if any(word in user_lower for word in ["history", "social", "geography"]):
                topics.append("Social Studies")

        # Extract basic facts from AI response
        if ai_response:
            ai_lower = ai_response.lower()
            if "answer" in ai_lower or "solution" in ai_lower:
                facts.append("Problem solving discussion")
            if "explain" in ai_lower or "understand" in ai_lower:
                facts.append("Concept explanation provided")

        # Create a more intelligent summary
        if topics:
            summary_text = (
                f"Educational conversation covering {', '.join(topics)} topics."
            )
        else:
            summary_text = "Educational conversation covering various academic topics."

        return {
            "updated_summary": summary_text,
            "key_topics": topics or ["General Discussion"],
            "recent_focus": "User is engaged in educational learning and problem-solving.",
            "accumulated_facts": facts or ["Learning in progress"],
            "question_numbers": [],
            "learning_progression": "Learning in progress",
            "difficulty_trend": "beginner",
            "learning_patterns": [],
            "struggling_areas": [],
            "mastered_concepts": [],
            "version": 1,
            "last_updated": datetime.now().isoformat(),
        }

    async def get_summary_version_history(
        self, interaction_id: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get version history of semantic summaries for an interaction

        DEPRECATED: ConversationContext table was removed in RAG streamlining.
        This function now returns an empty list for backwards compatibility.
        """
        print(
            f"⚠️ Summary version history unavailable (table removed in RAG streamlining)"
        )
        return []

    async def store_summary_version(
        self, interaction_id: str, user_id: str, summary_data: Dict[str, Any]
    ) -> bool:
        """
        Store a version of the semantic summary for history tracking

        DEPRECATED: ConversationContext table was removed in RAG streamlining.
        This function now returns True as a no-op for backwards compatibility.
        """
        print(f"⚠️ Summary version storage skipped (table removed in RAG streamlining)")
        return True

    def _generate_summary_hash(self, summary_data: Dict[str, Any]) -> str:
        """Generate a hash for the summary data"""
        import hashlib

        # Create a string representation of the summary
        summary_string = json.dumps(summary_data, sort_keys=True)
        return hashlib.md5(summary_string.encode()).hexdigest()

    async def analyze_summary_quality(
        self, summary_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze the quality of a semantic summary
        """
        try:
            quality_metrics = {
                "completeness_score": 0.0,
                "clarity_score": 0.0,
                "relevance_score": 0.0,
                "overall_score": 0.0,
                "issues": [],
                "recommendations": [],
            }

            # Check completeness
            required_fields = ["updated_summary", "key_topics", "accumulated_facts"]
            present_fields = sum(
                1 for field in required_fields if summary_data.get(field)
            )
            quality_metrics["completeness_score"] = present_fields / len(
                required_fields
            )

            if quality_metrics["completeness_score"] < 1.0:
                quality_metrics["issues"].append("Missing required fields")
                quality_metrics["recommendations"].append(
                    "Ensure all required fields are present"
                )

            # Check clarity (summary length and structure)
            summary_text = summary_data.get("updated_summary", "")
            if 50 <= len(summary_text) <= 500:
                quality_metrics["clarity_score"] = 1.0
            elif len(summary_text) < 50:
                quality_metrics["clarity_score"] = 0.5
                quality_metrics["issues"].append("Summary too short")
                quality_metrics["recommendations"].append(
                    "Add more detail to the summary"
                )
            else:
                quality_metrics["clarity_score"] = 0.7
                quality_metrics["issues"].append("Summary too long")
                quality_metrics["recommendations"].append("Compress the summary")

            # Check relevance (presence of educational content)
            educational_indicators = [
                "learn",
                "understand",
                "problem",
                "solution",
                "concept",
                "equation",
            ]
            summary_lower = summary_text.lower()
            educational_count = sum(
                1 for indicator in educational_indicators if indicator in summary_lower
            )
            quality_metrics["relevance_score"] = min(1.0, educational_count / 3)

            if quality_metrics["relevance_score"] < 0.5:
                quality_metrics["issues"].append("Low educational relevance")
                quality_metrics["recommendations"].append(
                    "Focus more on educational content"
                )

            # Calculate overall score
            quality_metrics["overall_score"] = (
                quality_metrics["completeness_score"]
                + quality_metrics["clarity_score"]
                + quality_metrics["relevance_score"]
            ) / 3

            return quality_metrics

        except Exception as e:
            print(f"Error analyzing summary quality: {e}")
            return {
                "completeness_score": 0.0,
                "clarity_score": 0.0,
                "relevance_score": 0.0,
                "overall_score": 0.0,
                "issues": ["Analysis failed"],
                "recommendations": ["Check summary format"],
            }


# Global instance
semantic_summary_service = SemanticSummaryService()
