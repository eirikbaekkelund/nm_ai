import json

from claude_client import CLAUDE_CLIENT, CLAUDE_MODEL
from utils import extract_tag


async def summarize_task(messages, memory_index):
    summary_prompt = f"""
    You are an assistant that will summarize how a Re-Act agent solved a task using the Tripletex API.
    The point is that the same Re-Act agent will be able to search for prior similar tasks in order to be as efficient as possible (e.g., reuse steps, avoid pitfalls, etc.).

    The Re-Act agent encounters tasks like:
    - Employees: Create employees, set roles, update contact info
    - Customers & Products: Register customers, create products
    - Invoicing: Create invoices, register payments, issue credit notes
    - Travel Expenses: Register or delete travel expense reports
    - Projects: Create projects linked to customers
    - Corrections: Delete or reverse incorrect entries
    - Departments: Create departments, enable accounting modules

    It's input can be in any language, but it (and you) should respond in English. 
    The input can also have files attached.

    You will be given the task input + all iterations of the agent's thoughts and actions (the conversation).
    The agent had these tools to solve the task:
    - get_endpoints_high_level_info: find relevant API endpoints, filtering by endpoint category and method (POST, GET, etc.). Use this to find out which endpoints are relevant.
    - get_endpoints_detailed_info: get detailed info of Tripletex API endpoints for certain operation IDs. Use this after you have identified relevant endpoints via get_endpoints_high_level_info and need more details to know how to call them.
    - get_component_schema: get detailed info of a endpoint component schema by name. Use this to resolve the structure of request bodies or response objects (e.g. for "schema": {{"$ref": "#/components/schemas/Voucher"}}, call get_component_schema(name="Voucher") to get the details of that schema).
    - get_faqs: see answer to relevant FAQs about the Tripletex API
    - call_tripletex_api: execute API requests

    Now, summarize how this task was solved.
    The Re-Act agent receives bonuses.
    Two factors determine the bonus:
    1. Call efficiency — How many write calls (POST, PUT, DELETE, PATCH) did the agent make? Fewer calls = higher bonus. GET requests are not counted — read as much as you need to understand the data.
    2. Error cleanliness — How many of the write calls resulted in 4xx errors (400, 404, 422, etc.)? Errors reduce the bonus. An agent that gets it right without trial-and-error is rewarded.

    You want to help the agent maximize its bonus in future similar tasks, so focus on insights that can help it be more efficient, fewer calls and make fewer write call errors.
    Therefore you **must** pay special attention to any calls that resulted in errors, and try to understand why the error happened and how to avoid it in the future.

        Task Input:
    {messages[0]["content"]}

    Agent actions and thoughts:
    {messages[1:]}

    Return JSON wrapped in <json> tags with this structure:
    <json>
    {{
    "task": "...",
    "steps": ["step1", "step2"],
    "key_endpoints": ["..."],
    "pitfalls": ["..."]
    }}
    </json>

    To avoid recreating summaries that already exist in memory, check the following provided memory index (list of prior tasks):
    
    {memory_index}

    If any of the above summaries covers the same task and has a good structure, there is no need to output a new summary, and instead you should output an empty json.
    
    # Rules:
    - Be concise
    - **THIS IS VERY IMPORTANT**: Do not include explicit names or numbers (e.g. customer names, project names, account codes, etc.) in the summary, as these are subject to change. 
    - Focus on how to avoid error for write calls (POST, PUT) and how to make minimal necessary write calls
    - No explanations
    - Output ONLY JSON
    - Output empty JSON ({{}}) if the task is already well covered by an existing summary in memory (based on the provided memory index)
    """

    response = await CLAUDE_CLIENT.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        temperature=0,
        messages=[{"role": "user", "content": summary_prompt}]
    )
    
    json_str = response.content[0].text
    json_text = extract_tag(json_str, "json")
    if not json_text:
        raise ValueError("⚠️ No JSON found in summary response")
        
    
    data = json.loads(json_text)
    if not data:
        print("⚠️ No JSON found in summary response, perhaps agent found that an existing summary was sufficient.")
        return
    
    print(data)
    print()
    store_summary(data)

    return data

    
def store_summary(summary, path="memory.json"):
    current_summaries: list[dict] = load_summaries(path)
    if not current_summaries:
        current_summaries = []
    summary['id'] = len(current_summaries) + 1
    current_summaries.append(summary)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(current_summaries, f, indent=4, ensure_ascii=False)

def load_summaries(path="memory.json") -> list[dict]:
    with open(path, 'r', encoding="utf-8") as f:
        return json.load(f)


def get_memory_index():
    memories = load_summaries()
    return [
        {"id": mem["id"], "task": mem["task"]}
        for mem in memories
    ]

async def get_memories(ids):
    memories = load_summaries()
    return [exp for exp in memories if exp["id"] in ids]