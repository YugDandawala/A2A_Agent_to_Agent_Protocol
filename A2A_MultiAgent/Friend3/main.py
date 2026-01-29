import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from Agent_Executor import Friend3Executor

def main():
    card = AgentCard(
        name="Friend3",
        description="Essay Writing Agent",
        url="http://localhost:10003",
        version="1.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[AgentSkill(
            id="essay",
            name="Generate essay",
            description="Writes and self-rates essay",
            tags=["essay", "writing", "content"],
            examples=["Create an essay from these outlines"]
        )]
    )

    handler = DefaultRequestHandler(
        agent_executor=Friend3Executor(),
        task_store=InMemoryTaskStore()
    )

    app = A2AStarletteApplication(agent_card=card, http_handler=handler)
    uvicorn.run(app.build(), host="0.0.0.0", port=10003)

if __name__ == "__main__":
    main()