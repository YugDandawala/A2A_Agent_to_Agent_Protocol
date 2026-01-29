from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.tasks import TaskUpdater
from a2a.server.events import EventQueue
from a2a.types import Part, TextPart,UnsupportedOperationError
from Agent import OutlineGraph
from a2a.utils.errors import ServerError

class Friend2Executor(AgentExecutor):
    def __init__(self):
        self.agent = OutlineGraph()

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        if not context.current_task:
            await updater.submit()

        await updater.start_work()

        research = context.get_user_input()
        outlines = await self.agent.run(research)

        message = f"{outlines}\n\n✅ Friend 2: My task is done. Thank you for providing me insightful knowledge."

        parts = [Part(root=TextPart(text=message))]

        await updater.add_artifact(parts, name="outlines")
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """
        Called if the task is cancelled by the client.
        """
        raise ServerError(error=UnsupportedOperationError())