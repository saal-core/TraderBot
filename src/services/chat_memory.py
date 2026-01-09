"""
Chat Memory Manager for maintaining conversation history
Works with both Streamlit UI and FastAPI endpoints
"""
from typing import List, Dict, Optional
from collections import deque


class ChatMemory:
    """
    Manages conversation history with a sliding window of recent messages
    Thread-safe and stateless for FastAPI usage
    """

    def __init__(self, max_pairs: int = 5):
        """
        Initialize chat memory

        Args:
            max_pairs: Maximum number of Q&A pairs to retain (default: 5)
        """
        self.max_pairs = max_pairs
        self.max_messages = max_pairs * 2  # Each pair = user + assistant message

    def add_message(self, history: List[Dict], role: str, content: str) -> List[Dict]:
        """
        Add a message to history and maintain the sliding window

        Args:
            history: Current conversation history
            role: Message role ("user" or "assistant")
            content: Message content

        Returns:
            Updated history with new message
        """
        # Create new message
        new_message = {
            "role": role,
            "content": content
        }

        # Add to history
        history.append(new_message)

        # Maintain sliding window (keep last N pairs)
        if len(history) > self.max_messages:
            # Remove oldest pair (2 messages)
            history = history[-self.max_messages:]

        return history

    def get_context_messages(self, history: List[Dict], include_system: bool = False) -> List[Dict]:
        """
        Get recent messages for context

        Args:
            history: Full conversation history
            include_system: Whether to include system messages

        Returns:
            List of recent messages for context
        """
        if not history:
            return []

        # Get last N messages (respecting max_pairs)
        recent_messages = history[-self.max_messages:]

        if not include_system:
            # Filter out system messages if any
            recent_messages = [msg for msg in recent_messages if msg.get("role") != "system"]

        return recent_messages

    def format_for_llm(self, history: List[Dict]) -> List[Dict]:
        """
        Format conversation history for LLM consumption

        Args:
            history: Conversation history

        Returns:
            Formatted messages for LLM
        """
        recent_messages = self.get_context_messages(history)

        # Ensure we have valid role-content pairs
        formatted = []
        for msg in recent_messages:
            if msg.get("role") and msg.get("content"):
                formatted.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        return formatted

    def get_summary_text(self, history: List[Dict]) -> str:
        """
        Get a text summary of recent conversation for prompt injection

        Args:
            history: Conversation history

        Returns:
            Formatted text summary
        """
        recent_messages = self.get_context_messages(history)

        if not recent_messages:
            return "No previous conversation."

        summary_parts = []
        for msg in recent_messages:
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")
            summary_parts.append(f"{role}: {content}")

        return "\n".join(summary_parts)

    def clear_history(self, history: List[Dict]) -> List[Dict]:
        """
        Clear conversation history

        Args:
            history: Current history

        Returns:
            Empty history list
        """
        return []

    def get_last_user_message(self, history: List[Dict]) -> Optional[str]:
        """
        Get the last user message from history

        Args:
            history: Conversation history

        Returns:
            Last user message content or None
        """
        for msg in reversed(history):
            if msg.get("role") == "user":
                return msg.get("content")
        return None

    def get_last_assistant_message(self, history: List[Dict]) -> Optional[str]:
        """
        Get the last assistant message from history

        Args:
            history: Conversation history

        Returns:
            Last assistant message content or None
        """
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                return msg.get("content")
        return None

    def count_pairs(self, history: List[Dict]) -> int:
        """
        Count the number of Q&A pairs in history

        Args:
            history: Conversation history

        Returns:
            Number of complete pairs
        """
        user_count = sum(1 for msg in history if msg.get("role") == "user")
        assistant_count = sum(1 for msg in history if msg.get("role") == "assistant")

        # Return the minimum to count complete pairs
        return min(user_count, assistant_count)


class SessionMemoryManager:
    """
    Manager for multiple chat sessions (useful for FastAPI with multiple users)
    """

    def __init__(self, max_pairs: int = 5):
        """
        Initialize session memory manager

        Args:
            max_pairs: Maximum Q&A pairs per session
        """
        self.max_pairs = max_pairs
        self.sessions: Dict[str, List[Dict]] = {}

    def get_session(self, session_id: str) -> List[Dict]:
        """
        Get or create a session history

        Args:
            session_id: Unique session identifier

        Returns:
            Session conversation history
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]

    def add_message(self, session_id: str, role: str, content: str) -> List[Dict]:
        """
        Add message to a specific session

        Args:
            session_id: Session identifier
            role: Message role
            content: Message content

        Returns:
            Updated session history
        """
        history = self.get_session(session_id)
        memory = ChatMemory(max_pairs=self.max_pairs)
        updated_history = memory.add_message(history, role, content)
        self.sessions[session_id] = updated_history
        return updated_history

    def clear_session(self, session_id: str):
        """
        Clear a specific session

        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            self.sessions[session_id] = []

    def delete_session(self, session_id: str):
        """
        Delete a session completely

        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_all_sessions(self) -> List[str]:
        """
        Get all active session IDs

        Returns:
            List of session identifiers
        """
        return list(self.sessions.keys())
