import json
import re

from pydantic import BaseModel


def resolve_ref(ref: str, schemas: dict):
    # ref like "#/components/schemas/EmploymentDetails"
    parts = ref.strip("#/").split("/")
    obj = {"components": {"schemas": schemas}}

    for part in parts:
        obj = obj[part]

    return obj


def resolve_schema(schema, schemas, depth=0, max_depth=0):
    if not isinstance(schema, dict):
        return schema

    if depth > max_depth:
        return {"type": "object"}

    if "$ref" in schema:
        resolved = resolve_ref(schema["$ref"], schemas)
        return resolve_schema(resolved, schemas, depth + 1, max_depth)

    if "properties" in schema:
        return {
            "type": "object",
            "properties": {
                k: resolve_schema(v, schemas, depth + 1, max_depth)
                for k, v in schema["properties"].items()
            },
        }

    if "items" in schema:
        return {
            "type": "array",
            "items": resolve_schema(schema["items"], schemas, depth + 1, max_depth),
        }

    return schema


def extract_schema(content: dict):
    if not content:
        return None

    # Try common content types first
    preferred_keys = ["application/json", "application/json; charset=utf-8"]

    for key in preferred_keys:
        if key in content and "schema" in content[key]:
            return content[key]["schema"]

    # Fallback: find ANY schema
    for key, value in content.items():
        if isinstance(value, dict) and "schema" in value:
            return value["schema"]

    return None


def parse_openapi(paths, schemas):
    endpoints = []

    for path, methods in paths.items():
        for method, details in methods.items():

            endpoint = {
                "operation_id": details.get("operationId"),
                "method": method.upper(),
                "path": path,
                "summary": details.get("summary", ""),
                "tags": details.get("tags", []),
                "parameters": [],
                "request_body": None,
                "response": None,
            }

            # --- Parameters ---
            for param in details.get("parameters", []):
                endpoint["parameters"].append(
                    {
                        "name": param["name"],
                        "in": param["in"],
                        "required": param.get("required", False),
                        "type": param.get("schema", {}).get("type"),
                        "description": param.get("description", "")
                    }
                )

            # --- Request Body ---
            if "requestBody" in details:
                content = details["requestBody"]["content"]

                # if "application/json" in content:
                #     schema = content["application/json"]["schema"]
                # else:
                #     # fallback (Tripletex uses charset variant)
                #     key = list(content.keys())[0]
                #     schema = content[key]["schema"]
                schema = extract_schema(content)
                endpoint['request_body'] = schema
                # if schema:
                #     endpoint["request_body"] = resolve_schema(schema, schemas)
                # else:
                #     print("⚠️ No schema found for:", endpoint["operation_id"])

            # --- Response (200 only for now) ---
            responses = details.get("responses", {})
            if "200" in responses:
                content = responses["200"].get("content", {})
                # if "application/json" in content:
                #     schema = content["application/json"]["schema"]
                #     endpoint["response"] = resolve_schema(schema, schemas)
                schema = extract_schema(content)
                if schema:
                    endpoint["response"] = resolve_schema(schema, schemas)
                else:
                    print("⚠️ No schema found for:", endpoint["operation_id"])

            endpoints.append(endpoint)

    return endpoints


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return text


class Endpoint(BaseModel):
    operation_id: str
    method: str
    path: str
    summary: str
    tags: list
    parameters: list
    request_body: dict | None = None
    response: dict | None = None

    def to_doc(self):
        return f"""
        Operation: {self.operation_id}
        Method: {self.method}
        Path: {self.path}
        Summary: {self.summary}
        Tags: {", ".join(self.tags)}
        Parameters: {", ".join(self.parameters)}
        Request: {self.request_body}
        Response: {self.response}
    """

    def high_level_info(self):
        return {
            "operation_id": self.operation_id,
            "method": self.method,
            "path": self.path,
            "summary": self.summary,
        }
    
    def detailed_info(self):
        return {
            "operation_id": self.operation_id,
            "method": self.method,
            "path": self.path,
            "summary": self.summary,
            "parameters": self.parameters,
            "request_body": self.request_body,
            "response": self.response,
        }

    def to_dict(self):
        return {
            "operation_id": self.operation_id,
            "method": self.method,
            "path": self.path,
            "summary": self.summary,
            "parameters": self.parameters,
            "request_body": self.request_body,
            "response": self.response,
        }

    @property
    def search_text(self):
        # return normalize(self.summary)
        return normalize(" ".join([self.summary, self.path]))
    
    def __hash__(self):
        return hash(self.operation_id)


def to_endpoint(ep):
    params = [f"{p['name']} ({p['in']}, {p['type']}, required={p['required']}, description: {p.get('description')})" for p in ep["parameters"]]

    # body_fields = []
    # if ep["request_body"] and "properties" in ep["request_body"]:
    #     for k, v in ep["request_body"]["properties"].items():
    #         body_fields.append(f"{k} ({v.get('type')})")

    # body_str = ", ".join(body_fields)

    return Endpoint(
        operation_id=ep["operation_id"],
        method=ep["method"],
        path=ep["path"],
        summary=ep["summary"],
        tags=ep["tags"],
        parameters=params,
        request_body=ep['request_body'],
        response=ep['response'],
    )


# Parse OpenAPI spec and save endpoints to file 
import sys

with open("tripletex_openapi.json") as f:
    spec = json.load(f)

paths = spec["paths"]
components = spec.get("components", {})
schemas = components.get("schemas", {})

endpoints_raw = parse_openapi(paths, schemas)

ENDPOINTS = [to_endpoint(ep) for ep in endpoints_raw]

import json 
with open("openapi/endpoints.json", "w") as f:
    json.dump([ep.model_dump() for ep in ENDPOINTS], f, indent=2)


with open("openapi/component_schemas.json", "w") as f:
    json.dump(components['schemas'], f, indent=2)


def get_component(name):
    return components['schemas'].get(name)
