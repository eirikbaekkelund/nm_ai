from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

app = FastAPI()

BASE_URL = None 
SESSION_TOKEN = None

class File(BaseModel):
    filename: str
    content_base64: str
    mime_type: str


class Credentials(BaseModel):
    base_url: str
    session_token: str


class TaskRequest(BaseModel):
    """
    prompt	string	The task in Norwegian natural language
    files	array	Attachments (PDFs, images) — may be empty
    files[].filename	string	Original filename
    files[].content_base64	string	Base64-encoded file content
    files[].mime_type	string	MIME type (application/pdf, image/png, etc.)
    tripletex_credentials.base_url	string	Proxy API URL — use this instead of the standard Tripletex URL
    tripletex_credentials.session_token	string	Session token for authentication
    """

    prompt: str
    files: Optional[List[File]] = []
    tripletex_credentials: Credentials


@app.get("/")
def read_root():
    return {"Hello": "World"}


from react import run_agent
from files import make_claude_file_block, parse_files
from memory import get_memory_index, summarize_task

@app.post("/solve")
async def solve(task: TaskRequest, background_tasks: BackgroundTasks):
    print("Received task:", task)

    global BASE_URL, SESSION_TOKEN
    BASE_URL = task.tripletex_credentials.base_url
    SESSION_TOKEN = task.tripletex_credentials.session_token

    # Get memory index
    MEMORY_INDEX = get_memory_index()

    # Parse files 
    files = parse_files(task.files)
    
    # files = [make_claude_file_block(f.content_base64, f.mime_type) for f in task.files]

    # Run agent 
    result = await run_agent(task.prompt, files, MEMORY_INDEX, BASE_URL, SESSION_TOKEN)
    print("Task solved:", result['task_solved'])

    # Save summary in background
    background_tasks.add_task(summarize_task, result["messages"], MEMORY_INDEX)
        
    return {"status": "completed"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
  