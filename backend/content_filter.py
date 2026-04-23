# CRITICAL — must run before every LLM call
# TCS blocks certain words — this catches them gracefully

BLOCKED_WORDS = [
    "horror", "extreme", "violence", "weapon", "kill",
    "suicide", "terror", "bomb", "explicit", "gore",
    "abuse", "illegal", "hack", "crack", "exploit"
]

def is_safe_query(query: str) -> tuple[bool, str]:
    """
    Check if the query contains blocked words.
    Returns (is_safe, message).
    """
    query_lower = query.lower()
    for word in BLOCKED_WORDS:
        if word in query_lower:
            return False, (
                f"Your query contains restricted content ('{word}'). "
                f"Please rephrase your question and try again."
            )
    return True, ""


def safe_llm_call(llm_func, *args, **kwargs):
    """
    Wrapper that catches LLM content policy errors gracefully.
    Use this to wrap every LLM call.
    """
    try:
        return llm_func(*args, **kwargs)
    except Exception as e:
        error_str = str(e).lower()
        if any(word in error_str for word in
               ["content filter", "policy", "blocked", "403", "safety"]):
            return (
                "I'm unable to process this request as it may contain "
                "restricted content. Please rephrase your query."
            )
        raise e