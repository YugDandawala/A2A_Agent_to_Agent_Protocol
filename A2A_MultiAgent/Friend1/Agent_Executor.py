from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.tasks import TaskUpdater
from a2a.server.events import EventQueue
from a2a.types import Part, TextPart, UnsupportedOperationError
from Agent import ResearchGraph
from a2a.utils.errors import ServerError

class Friend1Executor(AgentExecutor):
    def __init__(self):
        self.agent = ResearchGraph()

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        if not context.current_task:
            await updater.submit()

        await updater.start_work()

        topic = context.get_user_input()
        result = await self.agent.run(topic)

        parts = [Part(root=TextPart(text=result))]

        await updater.add_artifact(parts, name="research")
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """
        Called if the task is cancelled by the client.
        """
        raise ServerError(error=UnsupportedOperationError())