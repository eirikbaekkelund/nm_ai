
import re


def extract_tag(response: str, tag: str = "thinking") -> str:
    """Extracts reasoning from a response string based on a specified tag."""
    reasoning_match = re.search(f'<{tag}>(.*?)</{tag}>', response, flags=re.DOTALL | re.MULTILINE)
    reasoning = None

    if reasoning_match:
        # extract content between <thinking> tags
        reasoning = reasoning_match.group()
        reasoning = reasoning.replace(f"<{tag}>", "").replace(f"</{tag}>", "").strip("\n")

    return reasoning or ""