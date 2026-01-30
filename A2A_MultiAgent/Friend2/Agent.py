from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

llm = ChatGoogleGenerativeAI(
    model="Gemini 3 Flash",
    google_api_key="KEY",
    temperature=0.2,
)

class State(TypedDict):
    research: str
    outlines: List[str]
    count: int

async def generate_outlines(state: State):
    prompt = f"""
Analyze the research and produce outlines from it.
RESEARCH:
{state['research']}
"""
    res = await llm.ainvoke([HumanMessage(content=prompt)])

    lines = [l.strip("-• ") for l in res.content.split("\n") if l.strip()]
    return {"outlines": lines}

async def count_outlines(state: State):
    return {"count": len(state["outlines"])}

def router(state: State):
    if state["count"] < 5:
        return "retry"
    return "pass"

class OutlineGraph:
    def __init__(self):
        graph = StateGraph(State)

        graph.add_node("generate", generate_outlines)
        graph.add_node("count", count_outlines)

        graph.set_entry_point("generate")
        graph.add_edge("generate", "count")

        graph.add_conditional_edges("count", router, {"retry": "generate", "pass": END})

        self.graph = graph.compile()

    async def run(self, research: str):
        result = await self.graph.ainvoke({"research": research})
        outlines_text = "\n".join(f"- {o}" for o in result["outlines"])
        return outlines_text
