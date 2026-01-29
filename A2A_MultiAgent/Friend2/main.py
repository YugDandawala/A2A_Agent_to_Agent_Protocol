import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from Agent_Executor import Friend2Executor

def main():
    card = AgentCard(
        name="Friend2",
        description="Outline & Analysis Agent",
        url="http://localhost:10002",
        version="1.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[AgentSkill(
            id="outline",
            name="Generate outlines",
            description="Analyzes research and generates structured outlines",
            tags=["research", "analysis", "ai"],
            examples=["Analyze this research"]
        )]
    )

    handler = DefaultRequestHandler(
        agent_executor=Friend2Executor(),
        task_store=InMemoryTaskStore()
    )

    app = A2AStarletteApplication(agent_card=card, http_handler=handler)
    uvicorn.run(app.build(), host="0.0.0.0", port=10002)

if __name__ == "__main__":
    main()
