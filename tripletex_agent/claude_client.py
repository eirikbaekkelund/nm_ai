from anthropic import AsyncAnthropic
import os
from dotenv import load_dotenv


load_dotenv()
CLAUDE_CLIENT = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
CLAUDE_MODEL = "claude-opus-4-6"