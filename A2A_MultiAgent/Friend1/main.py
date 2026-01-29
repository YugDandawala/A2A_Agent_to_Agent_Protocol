import uvicorn, httpx
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from Agent_Executor import Friend1Executor

def main():
    card = AgentCard(
        name="Friend1",
        description="Research Agent",
        url="http://localhost:10001",
        version="1.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[AgentSkill(
            id="research",
            name="Research topic",
            description="Deep research with self-rating loop",
            tags=["research", "analysis", "ai"],
            examples=["AI in healthcare","AI in Electronics"]
        )]
    )

    handler = DefaultRequestHandler(
        agent_executor=Friend1Executor(),
        task_store=InMemoryTaskStore()
    )

    app = A2AStarletteApplication(agent_card=card, http_handler=handler)
    uvicorn.run(app.build(), host="0.0.0.0", port=10001)

if __name__ == "__main__":
    main()
