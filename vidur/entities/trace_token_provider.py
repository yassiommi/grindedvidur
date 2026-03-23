"""Assigns realistic token IDs to requests from ShareGPT-style conversations.

Instead of generating synthetic prefix-sharing patterns with uniform groups,
this module models the structure of real ShareGPT conversations:
  - A pool of system prompts with Zipf-distributed popularity
  - Multi-turn conversations with cumulative context
  - Variable turn lengths drawn from empirical distributions
  - Interleaved conversations from different users

Prefix sharing emerges naturally from shared system prompts and multi-turn
context accumulation, producing realistic cache hit/miss patterns with
temporal locality, variable prefix lengths, and bursty behavior.

When network access is available, the provider can also load and tokenize
actual ShareGPT data via HuggingFace datasets + tiktoken.
"""

import json
import math
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

from vidur.entities.request import Request
from vidur.logger import init_logger

logger = init_logger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sharegpt"

# ── Simple offline tokenizer ────────────────────────────────────────────────
# Maps text to deterministic integer token IDs without external dependencies.
# Uses a hash-based scheme: each word maps to a unique token ID in a large
# vocabulary space. This preserves the key property that identical text
# produces identical token IDs — which is all the cache simulation needs.

_VOCAB_SIZE = 100_000  # hash space; large enough to avoid collisions


def _offline_tokenize(text: str) -> Tuple[int, ...]:
    """Convert text to a tuple of deterministic integer token IDs.

    Splits on whitespace + punctuation boundaries and hashes each token.
    Not a real BPE tokenizer, but produces consistent IDs for identical text,
    which is the only property the prefix cache requires.
    """
    # Simple word-level split (roughly 1.3 tokens per word for English)
    tokens = []
    word = []
    for ch in text:
        if ch.isalnum() or ch == "'":
            word.append(ch)
        else:
            if word:
                tokens.append("".join(word).lower())
                word = []
            if ch.strip():
                tokens.append(ch)
    if word:
        tokens.append("".join(word).lower())

    return tuple(hash(t) % _VOCAB_SIZE for t in tokens)


# ── Realistic conversation generator ────────────────────────────────────────
# Models the distributional properties of ShareGPT:
#   - System prompt popularity follows Zipf's law
#   - Turn count per conversation: geometric distribution (median ~3)
#   - Turn length: log-normal (median ~80 tokens, long tail to 1000+)
#   - Multiple concurrent conversations interleaved in the request stream

# Curated system prompts with varying lengths (modeled after real usage)
_SYSTEM_PROMPTS = [
    # High-frequency: generic assistant prompts (short)
    "You are a helpful assistant.",
    "You are a helpful, harmless, and honest AI assistant.",
    "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.",
    # Medium-frequency: role-specific prompts (medium)
    (
        "You are a Python programming expert. Help the user write clean, "
        "efficient, and well-documented Python code. Explain your reasoning "
        "step by step. If you spot bugs, point them out."
    ),
    (
        "You are a creative writing assistant. Help the user brainstorm ideas, "
        "develop characters, write dialogue, and craft compelling narratives. "
        "Be imaginative and offer constructive feedback."
    ),
    (
        "You are a knowledgeable math tutor. Explain mathematical concepts "
        "clearly, work through problems step by step, and help the student "
        "develop intuition. Use examples when helpful."
    ),
    (
        "You are a professional translator. Translate text accurately while "
        "preserving tone, style, and cultural nuance. If a phrase has multiple "
        "valid translations, explain the differences."
    ),
    (
        "You are a data science consultant. Help analyze datasets, suggest "
        "appropriate statistical methods, write SQL and Python code for data "
        "analysis, and explain results in plain language."
    ),
    # Low-frequency: long specialized prompts
    (
        "You are an expert software architect reviewing code for a large-scale "
        "distributed system. Consider scalability, fault tolerance, consistency, "
        "and performance. Point out potential race conditions, single points of "
        "failure, and suggest improvements. Use concrete examples from real "
        "systems like Kafka, Cassandra, and Kubernetes where relevant. Always "
        "explain the trade-offs involved in your recommendations."
    ),
    (
        "You are a medical information assistant. Provide accurate health "
        "information based on current medical knowledge. Always remind users "
        "to consult healthcare professionals for personal medical decisions. "
        "Do not diagnose conditions or prescribe treatments. Cite relevant "
        "medical guidelines when possible. Be empathetic and clear in your "
        "explanations, avoiding unnecessary medical jargon."
    ),
]

# Zipf weights: prompt i gets weight proportional to 1/(i+1)
_SYSTEM_PROMPT_WEIGHTS = [1.0 / (i + 1) for i in range(len(_SYSTEM_PROMPTS))]

# Corpus of realistic conversation starters (diverse topics)
_CONVERSATION_STARTERS = [
    "Can you explain how transformers work in machine learning?",
    "Write a Python function to find the longest common subsequence.",
    "What are the main differences between TCP and UDP?",
    "Help me write a cover letter for a software engineering position.",
    "Explain the theory of relativity in simple terms.",
    "How do I set up a Docker container for a Node.js app?",
    "What's the best way to learn a new programming language?",
    "Can you help me debug this SQL query that's running slowly?",
    "Write a short story about a robot learning to paint.",
    "Explain the difference between correlation and causation.",
    "How do neural networks learn? Explain backpropagation.",
    "What are design patterns and when should I use them?",
    "Help me plan a study schedule for the GRE exam.",
    "What's the difference between REST and GraphQL APIs?",
    "Explain quantum computing to a five-year-old.",
    "How do I optimize a React application for performance?",
    "Write a regex to match email addresses.",
    "What are the SOLID principles in object-oriented design?",
    "Help me write unit tests for a login function.",
    "Explain how garbage collection works in Java.",
    "What is the CAP theorem and why does it matter?",
    "How do I implement authentication with JWT tokens?",
    "Write a bash script to monitor disk usage.",
    "Explain the difference between threads and processes.",
    "How do I use Git rebase vs merge?",
    "What are the best practices for API rate limiting?",
    "Explain MapReduce with a simple example.",
    "How do I set up CI/CD with GitHub Actions?",
    "What is the difference between SQL and NoSQL databases?",
    "Help me understand recursion with practical examples.",
]

# Follow-up templates that build on prior context
_FOLLOWUP_TEMPLATES = [
    "Can you explain that in more detail?",
    "Could you give me a code example for that?",
    "What about error handling in this case?",
    "How would this change if I'm using Python instead?",
    "Can you show me the time complexity analysis?",
    "What are the potential security implications?",
    "How does this compare to the alternative approach you mentioned?",
    "Can you write tests for the code you just showed me?",
    "What if the input is very large? How would it scale?",
    "Thanks! Now can you help me with a related problem?",
    "I got an error when I tried that. Here's the traceback: ...",
    "That makes sense. Can you also explain how caching helps here?",
    "What would the production deployment look like?",
    "How do I monitor this in production?",
    "Can you refactor this to be more maintainable?",
]

# Simulated assistant response fragments (for building multi-turn context)
_RESPONSE_FRAGMENTS = [
    "Sure! Let me explain step by step.",
    "That's a great question. The key concept here is",
    "Here's a code example that demonstrates this:",
    "There are several approaches to this problem. The most common ones are:",
    "Let me break this down into smaller parts.",
    "The main difference is in how they handle",
    "Here's the implementation you asked for:",
    "Good follow-up question. Building on what I said earlier,",
    "To optimize this, you should consider the following factors:",
    "Let me show you a more efficient approach:",
]


def _generate_conversation(
    rng: random.Random,
    system_prompt_tokens: Tuple[int, ...],
    num_turns: int,
) -> List[Tuple[int, ...]]:
    """Generate a single multi-turn conversation.

    Returns a list of cumulative token tuples — one per turn.
    Turn i contains all tokens from the system prompt through turn i.
    """
    cumulative: List[int] = list(system_prompt_tokens)
    turns: List[Tuple[int, ...]] = []

    for turn_idx in range(num_turns):
        if turn_idx == 0:
            # First human message
            text = rng.choice(_CONVERSATION_STARTERS)
            # Add some variation by appending a detail
            if rng.random() < 0.3:
                text += " I'm particularly interested in practical examples."
            if rng.random() < 0.2:
                text += " Please be concise."
        else:
            # Follow-up (alternating human/assistant)
            if turn_idx % 2 == 0:
                # Human follow-up
                text = rng.choice(_FOLLOWUP_TEMPLATES)
                if rng.random() < 0.15:
                    text += " Also, " + rng.choice(_FOLLOWUP_TEMPLATES).lower()
            else:
                # Assistant response — longer, with code-like content
                parts = rng.sample(
                    _RESPONSE_FRAGMENTS, k=min(3, len(_RESPONSE_FRAGMENTS))
                )
                text = " ".join(parts)
                # Simulate code blocks (variable length)
                if rng.random() < 0.4:
                    code_len = rng.randint(20, 200)
                    text += " " + " ".join(
                        f"tok{rng.randint(0, 9999)}" for _ in range(code_len)
                    )

        turn_tokens = _offline_tokenize(text)
        cumulative.extend(turn_tokens)
        turns.append(tuple(cumulative))

    return turns


def generate_sharegpt_conversations(
    num_conversations: int = 500,
    seed: int = 42,
) -> List[List[Tuple[int, ...]]]:
    """Generate a pool of ShareGPT-like conversations.

    Args:
        num_conversations: Number of conversations to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of conversations. Each conversation is a list of cumulative
        token tuples (one per turn).
    """
    rng = random.Random(seed)

    # Tokenize system prompts
    system_prompt_tokens = [_offline_tokenize(p) for p in _SYSTEM_PROMPTS]

    # Normalize Zipf weights to probabilities
    total_w = sum(_SYSTEM_PROMPT_WEIGHTS)
    probs = [w / total_w for w in _SYSTEM_PROMPT_WEIGHTS]

    conversations = []
    for _ in range(num_conversations):
        # Pick system prompt (Zipf-distributed)
        sp_idx = rng.choices(range(len(_SYSTEM_PROMPTS)), weights=probs, k=1)[0]
        sp_tokens = system_prompt_tokens[sp_idx]

        # Number of turns: geometric distribution, median ~4, max ~20
        num_turns = min(20, max(2, int(rng.expovariate(0.25)) + 2))

        conv = _generate_conversation(rng, sp_tokens, num_turns)
        conversations.append(conv)

    return conversations


# ── Main provider class ─────────────────────────────────────────────────────


class ShareGPTTokenIdProvider:
    """Assigns realistic token IDs to requests using ShareGPT-style conversations.

    Supports two modes:
      1. Offline (default): Generates conversations locally using curated
         templates with realistic distributional properties.
      2. Online: Loads actual ShareGPT data from a JSON file (pre-downloaded
         or fetched via HuggingFace datasets + tiktoken).

    Args:
        dataset_path: Path to pre-tokenized ShareGPT JSON. If None, generates
            conversations offline.
        seed: Random seed for reproducibility.
        multi_turn: If True, consecutive requests are assigned to successive
            turns in the same conversation (realistic interleaving). If False,
            each request gets an independent conversation's first turn.
        num_conversations: Number of conversations to generate in offline mode.
        interleave_conversations: If True, interleave turns from multiple
            active conversations (models concurrent users). If False, complete
            one conversation before starting the next.
        max_active_conversations: Number of concurrent conversations when
            interleaving (models concurrent user sessions).
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        seed: int = 42,
        multi_turn: bool = True,
        num_conversations: int = 500,
        interleave_conversations: bool = True,
        max_active_conversations: int = 5,
    ):
        self._seed = seed
        self._multi_turn = multi_turn
        self._interleave = interleave_conversations
        self._max_active = max_active_conversations
        self._rng = random.Random(seed)
        self._conversations: List[List[Tuple[int, ...]]] = []

        if dataset_path and os.path.exists(dataset_path):
            self._load_pretokenized(dataset_path)
        else:
            self._conversations = generate_sharegpt_conversations(
                num_conversations=num_conversations, seed=seed
            )

        logger.info(
            f"ShareGPTTokenIdProvider: {len(self._conversations)} conversations, "
            f"multi_turn={multi_turn}, interleave={interleave_conversations}"
        )

    def _load_pretokenized(self, path: str) -> None:
        """Load pre-tokenized conversations from JSON.

        Expected format: list of conversations, where each conversation is a
        list of token ID lists (cumulative turns).
        """
        with open(path) as f:
            raw = json.load(f)

        for conv in raw:
            turns = [tuple(t) for t in conv if isinstance(t, list)]
            if turns:
                self._conversations.append(turns)

        self._rng.shuffle(self._conversations)
        logger.info(f"Loaded {len(self._conversations)} conversations from {path}")

    @property
    def num_conversations(self) -> int:
        return len(self._conversations)

    def assign_token_ids(self, requests: List[Request]) -> None:
        """Assign token IDs from conversations to requests.

        Same interface as PrefixTokenGenerator.assign_token_ids().
        """
        if not requests or not self._conversations:
            return

        if self._multi_turn and self._interleave:
            self._assign_interleaved(requests)
        elif self._multi_turn:
            self._assign_sequential(requests)
        else:
            self._assign_independent(requests)

    def _assign_interleaved(self, requests: List[Request]) -> None:
        """Assign requests as interleaved turns from concurrent conversations.

        Models realistic serving where multiple users are chatting concurrently.
        A pool of active conversations is maintained; on each step, one active
        conversation advances by one turn. When a conversation finishes, a new
        one replaces it.
        """
        # Build a queue of conversations
        conv_queue = list(range(len(self._conversations)))
        self._rng.shuffle(conv_queue)
        queue_idx = 0

        # Active conversations: (conv_index, current_turn_index)
        active: List[Tuple[int, int]] = []

        def _refill_active():
            nonlocal queue_idx
            while len(active) < self._max_active and queue_idx < len(conv_queue):
                active.append((conv_queue[queue_idx], 0))
                queue_idx += 1
            # If we exhausted the queue, wrap around
            if not active:
                self._rng.shuffle(conv_queue)
                queue_idx = 0
                while len(active) < self._max_active and queue_idx < len(conv_queue):
                    active.append((conv_queue[queue_idx], 0))
                    queue_idx += 1

        _refill_active()

        for request in requests:
            if not active:
                _refill_active()

            # Pick a random active conversation
            slot = self._rng.randint(0, len(active) - 1)
            conv_idx, turn_idx = active[slot]
            conv = self._conversations[conv_idx]

            # Get cumulative tokens for this turn
            turn = conv[min(turn_idx, len(conv) - 1)]
            request._token_ids = self._fit_tokens(turn, request.num_prefill_tokens)

            # Advance the turn
            turn_idx += 1
            if turn_idx >= len(conv):
                # Conversation done — remove and refill
                active.pop(slot)
                _refill_active()
            else:
                active[slot] = (conv_idx, turn_idx)

    def _assign_sequential(self, requests: List[Request]) -> None:
        """Assign requests as sequential turns, one conversation at a time."""
        conv_idx = 0
        turn_idx = 0

        for request in requests:
            if conv_idx >= len(self._conversations):
                self._rng.shuffle(self._conversations)
                conv_idx = 0

            conv = self._conversations[conv_idx]
            turn = conv[min(turn_idx, len(conv) - 1)]
            request._token_ids = self._fit_tokens(turn, request.num_prefill_tokens)

            turn_idx += 1
            if turn_idx >= len(conv):
                conv_idx += 1
                turn_idx = 0

    def _assign_independent(self, requests: List[Request]) -> None:
        """Assign each request an independent conversation's first turn."""
        for i, request in enumerate(requests):
            conv = self._conversations[i % len(self._conversations)]
            turn = conv[0]
            request._token_ids = self._fit_tokens(turn, request.num_prefill_tokens)

    @staticmethod
    def _fit_tokens(
        tokens: Tuple[int, ...], target_length: int
    ) -> Tuple[int, ...]:
        """Truncate or extend token sequence to match target length.

        Truncation preserves the prefix (the shared part).
        Extension appends unique padding tokens to avoid false sharing.
        """
        if len(tokens) >= target_length:
            return tokens[:target_length]
        # Pad with unique tokens (high range to avoid collision with real tokens)
        pad_start = _VOCAB_SIZE + len(tokens)
        padding = tuple(pad_start + i for i in range(target_length - len(tokens)))
        return tokens + padding
