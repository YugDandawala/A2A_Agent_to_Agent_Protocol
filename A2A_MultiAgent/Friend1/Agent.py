from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model = "gemini-flash-latest",
    google_api_key = api_key,
    temperature = 0.4,
)

class State(TypedDict):
    topic: str
    research: str
    score: int

async def research(state: State):
    res = await llm.ainvoke([HumanMessage(content = f"Research on: {state['topic']}")])
    return {"research": res.content}

async def rate(state: State):
    res = await llm.ainvoke(
        [HumanMessage(content = f"Rate this research out of 10. Return a Number only:\n{state['research']}")]
    )
    score = int("".join(filter(str.isdigit, res.content)) or 0)
    return {"score": score}

def route(state: State):
    return "retry" if state["score"] < 7 else "done"

class ResearchGraph:
    def __init__(self):
        g = StateGraph(State)
        g.add_node("research", research)
        g.add_node("rate", rate)
        g.set_entry_point("research")
        g.add_edge("research", "rate")
        g.add_conditional_edges("rate", route, {"retry": "research", "done": END})
        self.graph = g.compile()

    async def run(self, topic: str):
        result = await self.graph.ainvoke({"topic": topic})
        return result["research"]