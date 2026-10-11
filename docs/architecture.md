
flowchart TD
    U[User Request] --> S[Supervisor Agent]
    S --> R[Worker Selection]
    R --> M[Math ReAct Worker]
    R --> W[Research Worker]
    R --> G[General Worker]

    M --> T[Tool Registry]
    T --> V[Pydantic Validation]
    V --> E[Tool Execution]
    E --> O[Tool Observation]
    O --> M

    M --> X[Worker Result]
    W --> X
    G --> X
    X --> F[Supervisor Synthesis]
    F --> A[Final Answer]

    M <--> H[Conversation Memory]
    H --> P[Context Pruning and Summary]
    H <--> C[(ChromaDB Long-Term Memory)]

    E --> L[Logging and Error Recovery]
    S --> L
