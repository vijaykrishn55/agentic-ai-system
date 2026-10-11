
import json

from groq import Groq

from src.config import GROQ_API_KEY, GROQ_MODEL
from src.tools.registry import create_default_registry


def run_function_calling(user_message: str) -> dict:
    client = Groq(api_key=GROQ_API_KEY)
    registry = create_default_registry()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use the available tools "
                "for arithmetic. Do not invent tool results."
            ),
        },
        {"role": "user", "content": user_message},
    ]

    # Allow multiple tool calls, but prevent infinite loops.
    for _ in range(5):
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=registry.get_schemas(),
            tool_choice="auto",
        )

        message = response.choices[0].message

        # No tool call: the model has produced its final answer.
        if not message.tool_calls:
            return {
                "success": True,
                "answer": message.content or "",
                "messages": messages,
            }

        messages.append(message.model_dump(exclude_none=True))

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name

            try:
                arguments = json.loads(tool_call.function.arguments)
                result = registry.execute(tool_name, arguments)
            except (json.JSONDecodeError, TypeError):
                result = {
                    "success": False,
                    "error": {
                        "type": "invalid_json",
                        "message": "Tool arguments were not valid JSON.",
                    },
                }

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    return {
        "success": False,
        "error": "Maximum tool-calling iterations reached.",
        "messages": messages,
    }


if __name__ == "__main__":
    query = input("Ask a question: ")
    output = run_function_calling(query)

    print(json.dumps(output, indent=2, default=str))
