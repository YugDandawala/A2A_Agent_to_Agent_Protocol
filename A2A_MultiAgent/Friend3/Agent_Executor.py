from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.tasks import TaskUpdater
from a2a.server.events import EventQueue
from a2a.types import Part, TextPart,UnsupportedOperationError
from Agent import EssayGraph
from a2a.utils.errors import ServerError

class Friend3Executor(AgentExecutor):
    def __init__(self):
        self.agent = EssayGraph()

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        if not context.current_task:
            await updater.submit()

        await updater.start_work()

        outlines = context.get_user_input()
        essay = await self.agent.run(outlines)

        message = f"{essay}\n\n✅ Friend 3: My task is done. Thank you for providing me insightful knowledge."

        parts = [Part(root=TextPart(text=message))]

        await updater.add_artifact(parts, name="final_essay")
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """
        Called if the task is cancelled by the client.
        """
        raise ServerError(error=UnsupportedOperationError())