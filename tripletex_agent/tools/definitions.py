

TOOLS = [
    {
        "name": "get_memories",
        "description": "Get relevant past memories from similar tasks. Use this to get inspiration and context from how similar tasks were solved in the past. Filter using list of memory ids.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": {"type": "integer"}
                },
            },
            "required": ["ids"]
        }
    },
    {
        "name": "get_faqs",
        "description": "Get answers to relevant FAQs about the Tripletex API. Filter using category (invoice, product/inventory, etc) and/or FAQ ids.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "ids": {
                    "type": "array",
                    "items": {"type": "integer"}
                },
            },
            "required": ["category"]
        }
    },
    
    {
        "name": "get_endpoints_high_level_info",
        "description": "Get high level info of Tripletex API endpoints for certain categories. Also possible to filter via list of `operation_id`s.",
        "input_schema": {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                
                "operation_ids": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "count": {"type": "integer", "default": 50},
                "offset": {"type": "integer", "default": 0}
            },
            "required": ["categories"]
        }
    },
    {
        "name": "get_endpoints_detailed_info",
        "description": "Get detailed info of Tripletex API endpoints for certain operation IDs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation_ids": {
                    "type": "array",
                    "items": {"type": "string"}
                },
            },
            "required": ["operation_ids"]
        }
    },
    {
        "name": "get_component_schema",
        "description": "Get detailed info of a Tripletex API component schema by name. Use this to get the structure of request bodies or responses.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"]
        }
    },
    {
        "name": "call_tripletex_api",
        "description": "Execute a Tripletex API request",
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {"type": "string", "description": "HTTP method (GET, POST, PUT, DELETE)"},
                "path": {"type": "string", "description": "API path, e.g., /customer, do not include v2 or any base part."},
                "path_params": {"type": "object"},
                "query_params": {"type": "object"},
                "body": {"type": "object"}
            },
            "required": ["method", "path"]
        }
    }
]