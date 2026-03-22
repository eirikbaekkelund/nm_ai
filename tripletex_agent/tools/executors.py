import asyncio
from typing import List, Tuple

import requests

from openapi.parse import Endpoint


def call_tripletex_api(base_url: str, session_token: str, action: dict) -> dict:
    path = action["path"]

    # replace path params
    for k, v in action.get("path_params", {}).items():
        path = path.replace(f"{{{k}}}", str(v))

    body = action.get("body")
    # Handle stringified JSON from LLM
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            pass

    response = requests.request(
        method=action["method"],
        url=base_url + path,
        auth=("0", session_token),
        params=action.get("query_params"),
        json=body
    )

    return {
        "action": action,
        "status_code": response.status_code,
        "response": response.json() if response.content else None
    }

class FAQ:
    def __init__(self, id: int, question: str, answer: str, category: str):
        self.id = id
        self.question = question
        self.answer = answer
        self.category = category
    
    def to_dict(self):
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer
        }
    

import json
with open("openapi/endpoints.json") as f:
    endpoints_data = json.load(f)

ENDPOINTS = [Endpoint(**ep) for ep in endpoints_data]

with open("openapi/component_schemas.json") as f:
    COMPONENT_SCHEMAS = json.load(f)

import json
with open("openapi/faqs.json") as f:
    faqs_data = json.load(f)

FAQS: List[FAQ] = []
for category, faqs in faqs_data.items():
    for faq in faqs:
        FAQS.append(FAQ(id=faq["id"], question=faq["question"], answer=faq["answer"], category=category))

def get_faqs_info():
    with open("openapi/faqs.json") as f:
        faqs_data = json.load(f)
    prompt = ""
    for category, faqs in faqs_data.items():
        prompt += f"## Category: {category}\n"
        for faq in faqs:
            prompt += f"(id={faq['id']}): {faq['question']}\n"

    return prompt


async def get_endpoint_categories():
    categories = set()
    for ep in ENDPOINTS:
        for tag in ep.tags:
            tag = __make_category(tag)
            categories.add(tag)

    return sorted(list(categories))

# cats = asyncio.run(get_endpoint_categories())
# print(cats)
def __make_category(tag):
    if tag.startswith("ledger/"):
        return tag.split('/')[0] + tag.split('/')[1].capitalize() + ''.join(tag.split('/')[2:])
    return tag.split('/')[0]

async def get_endpoints_high_level_info(categories: List[str], operation_ids: List[str] = None, count=50, offset=0):
    filtered: set[Endpoint] = set()

    for ep in ENDPOINTS:
        ep_cats = [__make_category(tag) for tag in ep.tags]

        if operation_ids is not None and ep.operation_id in operation_ids:
            filtered.add(ep)
        
        if any(cat in categories for cat in ep_cats):
            filtered.add(ep)

    return [ep.high_level_info() for ep in list(filtered)[offset:offset + count]]

async def get_endpoints_detailed_info(operation_ids: List[str]):
    filtered = []
    for ep in ENDPOINTS:
        if ep.operation_id in operation_ids:
            filtered.append(ep.detailed_info())

    return filtered

async def get_component_schema(name: str):
    return COMPONENT_SCHEMAS.get(name)

async def get_faqs(category: str, ids: List[int] = None):
    filtered = []
    for faq in FAQS:
        if faq.category == category and (ids is None or faq.id in ids):
            filtered.append(faq.to_dict())

    return filtered


# import base64
# import csv
# import io

# def parse_csv_file(file_obj):
#     raw = base64.b64decode(file_obj["content_base64"])
#     text = raw.decode("utf-8")

#     reader = csv.DictReader(io.StringIO(text), delimiter=";")

#     rows = [row for row in reader]
#     return rows

# file_obj = {'content_base64': 'RGF0bztGb3JrbGFyaW5nO0lubjtVdDtTYWxkbw0KMjAyNi0wMS0xNjtJbm5iZXRhbGluZyBmcmEgVGF5bG9yIEx0ZCAvIEZha3R1cmEgMTAwMTs2NTAwLjAwOzsxMDY1MDAuMDANCjIwMjYtMDEtMTc7SW5uYmV0YWxpbmcgZnJhIEpvaG5zb24gTHRkIC8gRmFrdHVyYSAxMDAyOzkwOTMuNzU7OzExNTU5My43NQ0KMjAyNi0wMS0xOTtJbm5iZXRhbGluZyBmcmEgU21pdGggTHRkIC8gRmFrdHVyYSAxMDAzOzE0MTg3LjUwOzsxMjk3ODEuMjUNCjIwMjYtMDEtMjE7SW5uYmV0YWxpbmcgZnJhIExld2lzIEx0ZCAvIEZha3R1cmEgMTAwNDs5Mzc1LjAwOzsxMzkxNTYuMjUNCjIwMjYtMDEtMjQ7SW5uYmV0YWxpbmcgZnJhIEpvaG5zb24gTHRkIC8gRmFrdHVyYSAxMDA1OzE2ODEyLjUwOzsxNTU5NjguNzUNCjIwMjYtMDEtMjY7QmV0YWxpbmcgU3VwcGxpZXIgV2lsbGlhbXMgTHRkOzstMTk0MDAuMDA7MTM2NTY4Ljc1DQoyMDI2LTAxLTI5O0JldGFsaW5nIFN1cHBsaWVyIFdpbHNvbiBMdGQ7Oy0xMzkwMC4wMDsxMjI2NjguNzUNCjIwMjYtMDItMDE7QmV0YWxpbmcgU3VwcGxpZXIgSGFycmlzIEx0ZDs7LTEzMjAwLjAwOzEwOTQ2OC43NQ0KMjAyNi0wMi0wMjtTa2F0dGV0cmVrazs7LTE1NjkuNTI7MTA3ODk5LjIzDQoyMDI2LTAyLTAzO0JhbmtnZWJ5cjs3NzUuNTE7OzEwODY3NC43NA0KMjAyNi0wMi0wNTtSZW50ZWlubnRla3Rlcjs7LTE5MDEuMzI7MTA2NzczLjQyDQo='}

# rows = parse_csv_file(file_obj)
# print(rows)
