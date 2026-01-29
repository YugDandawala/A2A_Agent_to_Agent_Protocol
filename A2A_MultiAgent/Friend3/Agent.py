from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

llm = ChatGoogleGenerativeAI(
    model="Gemini 2.5 Flash Lite",
    google_api_key="AIzaSyCgpXQGnJjpVwNHqot4GUa5Ho2n5PcfITc",
    temperature=0.4,
)

class State(TypedDict):
    outlines: str
    essay: str
    score: int

async def write_essay(state: State):
    prompt = f"""
Write an essay based on the outlines below:

{state['outlines']}
"""
    res = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"essay": res.content}

async def rate_essay(state: State):
    prompt = f"""
Rate the essay out of 10.
Example(6 out of 10,8/10)
ESSAY:
{state['essay']}
"""
    res = await llm.ainvoke([HumanMessage(content=prompt)])
    score = int("".join(filter(str.isdigit, res.content)) or 0)
    return {"score": score}

def router(state: State):
    if state["score"] < 7:
        return "retry"
    return "pass"

class EssayGraph:
    def __init__(self):
        graph = StateGraph(State)

        graph.add_node("write", write_essay)
        graph.add_node("rate", rate_essay)

        graph.set_entry_point("write")
        graph.add_edge("write", "rate")

        graph.add_conditional_edges("rate", router, {"retry": "write", "pass": END})

        self.graph = graph.compile()

    async def run(self, outlines: str):
        result = await self.graph.ainvoke({"outlines": outlines})
        return result["essay"]