import asyncio
import json

import anthropic

from memory import get_memories
from prompt import REACT_AGENT_SYSTEM_PROMPT
from tools.definitions import TOOLS
from claude_client import CLAUDE_CLIENT, CLAUDE_MODEL
from tools.executors import call_tripletex_api, get_endpoints_high_level_info, get_endpoints_detailed_info, get_component_schema, get_endpoint_categories, get_faqs, get_faqs_info
from datetime import datetime

async def run_agent(prompt, files, memory_index, base_url, token):
    ENDPOINT_CATEGORIES = await get_endpoint_categories()
    FAQS_INFO = get_faqs_info()
    
    if files: 
        prompt += "\n\nAttached files:\n"
        for f in files:
            prompt += f"{f['filename']}\n"
            prompt += f"Type: {f['type']}\n"
            prompt += f"Data: {f['data']}\n"

        print('##'*50)
        print(prompt)
        print('##'*50)


    messages = [
        {"role": "user", "content": prompt}
    ]

    task_solved = False
    max_iterations = 50
    iteration = 0
    print()
    while True:
        if iteration >= max_iterations:
            break
        iteration += 1

        success = False
        max_tries = 3
        tries = 0
        while not success and tries < max_tries:
            try:
                response = await CLAUDE_CLIENT.messages.create(
                    model=CLAUDE_MODEL,
                    system=REACT_AGENT_SYSTEM_PROMPT.format(
                        endpoint_categories=','.join(ENDPOINT_CATEGORIES), 
                        current_date=datetime.now().date(), 
                        faqs=FAQS_INFO, 
                        memory_index=memory_index
                    ),
                    max_tokens=1000,
                    temperature=0,
                    tools=TOOLS,
                    messages=messages,
                )
                success = True
            except anthropic._exceptions.OverloadedError:
                print("Claude is overloaded. Retrying in 10 seconds...")
                await asyncio.sleep(10)
                tries += 1

        messages.append({
            "role": "assistant",
            "content": response.content
        })

        tool_used = False

        # 🔁 Iterate over all blocks
        for block in response.content:
            print()
            if block.type == "text":
                print(block.text)

            elif block.type == "tool_use":
                tool_used = True

                tool_name = block.name
                tool_input = block.input

                print(f'Calling `{tool_name}` with input: {tool_input}')
                print()
                try:
                    if tool_name == "get_memories":
                        result = await get_memories(**tool_input)
                        
                    if tool_name == "get_faqs":
                        result = await get_faqs(**tool_input)

                    elif tool_name == "get_endpoints_high_level_info":
                        result = await get_endpoints_high_level_info(**tool_input)

                    elif tool_name == "get_endpoints_detailed_info":
                        result = await get_endpoints_detailed_info(**tool_input)

                    elif tool_name == "get_component_schema":
                        result = await get_component_schema(**tool_input)

                    elif tool_name == "call_tripletex_api":
                        result = call_tripletex_api(
                            base_url,
                            token,
                            tool_input
                        )
                except Exception as e:
                    result = {"error": str(e)}

                print(result)


                # ✅ Send tool result back correctly
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result)
                        }
                    ]
                })

        # ✅ If no tool was used → final answer
        if not tool_used:
            text_blocks = [
                b.text for b in response.content
                if b.type == "text"
            ]

            full_text = "\n".join(text_blocks)

            if '"status": "completed"' in full_text:
                task_solved = True
                break

            messages.append({
                "role": "user",
                "content": "If the task is complete, respond with {'status': 'completed'}"
            })

    return {"messages": messages, "task_solved": task_solved}