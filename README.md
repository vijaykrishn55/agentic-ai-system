
# Agentic AI System

A Python-based agentic AI system built with Groq's `openai/gpt-oss-120b`. It demonstrates function calling, a ReAct execution loop, conversation memory, semantic retrieval, multi-agent orchestration, and error recovery.

## Features

- **Function calling:** JSON Schema generation, Pydantic validation, tool registry, and dynamic execution.
- **ReAct agent:** Iterative tool calls, observations, execution history, and iteration limits.
- **Memory management:** Working memory, conversation history, sliding window, and bounded summary buffer.
- **Long-term memory:** ChromaDB-based semantic retrieval using local embeddings.
- **Multi-agent workflow:** Supervisor routes tasks to math, research, or general-purpose workers.
- **Error recovery:** Structured error observations and bounded retries.
- **Testing:** pytest tests for tools, memory, schemas, ReAct execution, recovery, and agent orchestration.

## Tech Stack

Python 3.11, Groq API, Pydantic, ChromaDB, python-dotenv, and pytest.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
LOG_LEVEL=INFO
```

Keep `.env` out of version control.

## Run

```bash
python -m src.tools.registry
python -m src.agents.function_caller
python -m src.agents.react_agent
python -m src.agents.multi_agent
```

The Groq-backed agents require a valid API key and available model quota. ChromaDB stores long-term memory locally.

## Tests

Run the complete test suite:

```bash
python -m pytest tests/ -v
```

The tests cover tool validation, generated schemas, memory operations, semantic retrieval, ReAct execution, error recovery, and supervisor delegation. Mocked tests are used where practical to avoid unnecessary API calls.

## Architecture

The request is routed through the supervisor or sent directly to the ReAct agent. The agent can call registered tools, observe their results, and continue until it produces an answer or reaches its execution limit. Conversation memory manages recent context, while ChromaDB stores and retrieves longer-term memories.

See [`docs/architecture.md`](docs/architecture.md) for the architecture diagram.

## Current Limitations

- The research worker does not use an external search API.
- The supervisor currently delegates each request to a single worker.
- The summary buffer truncates older text rather than generating intelligent summaries.
- Local semantic embeddings may require a model download on first use.
- The system has bounded retries, but production deployment would require additional API monitoring, stronger security controls, and more extensive integration testing.

## Future Improvements

Add external search tools, parallel worker execution, intelligent memory summarization, persistent conversation sessions, and deployment monitoring.

## License

Add a license before distributing the project if you intend to permit reuse.
