import asyncio
import uuid
import httpx
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from a2a.client import A2AClient, A2ACardResolver
from a2a.types import SendMessageRequest, MessageSendParams


# ==============================
# Host LLM (only for host work)
# ==============================

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key="AIzaSyCgpXQGnJjpVwNHqot4GUa5Ho2n5PcfITc",
    temperature=0.1,
)

# ==============================
# A2A Communication Helper
# ==============================


class A2AHostClient:
    def __init__(self):
        self.http = httpx.AsyncClient(timeout=60)

    async def connect(self, url: str) -> A2AClient:
        resolver = A2ACardResolver(self.http, url)
        card = await resolver.get_agent_card()
        return A2AClient(self.http, card, url=url)

    async def send(self, client: A2AClient, text: str) -> str:
        req = SendMessageRequest(
            id=str(uuid.uuid4()),
            params=MessageSendParams.model_validate(
                {
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": text}],
                        "messageId": str(uuid.uuid4()),
                    }
                }
            ),
        )

        resp = await client.send_message(req)

        if hasattr(resp.root, "error") and resp.root.error is not None:
            print("❌ Error from friend agent:", resp.root.error)
            return f"[ERROR]: {resp.root.error}"

        # ✅ Safely get the text if response succeeded
        result = getattr(resp.root, "result", None)
        if result is None or not result.artifacts:
            return "[ERROR]: No artifacts returned"
        try:
            return result.artifacts[0].parts[0].text
        except Exception as e:
            return f"[ERROR]: Malformed response: {e}"


# ==============================
# Host LangGraph State
# ==============================


class HostState(TypedDict):
    essay: str
    insights: str
    rating: int


# ==============================
# Host LangGraph Nodes
# ==============================


async def extract_insights(state: HostState):
    prompt = f"""
Extract the BEST and MOST IMPORTANT insights from this essay.

ESSAY:
{state['essay']}
"""
    res = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"insights": res.content}


async def rate_insights(state: HostState):
    prompt = f"""
Rate the quality of these insights out of 10.
Example(8 out of 10,8/10)

INSIGHTS:
{state['insights']}
"""
    res = await llm.ainvoke([HumanMessage(content=prompt)])
    score = int("".join(filter(str.isdigit, res.content)) or 0)
    return {"rating": score}


# ==============================
# Host LangGraph Brain
# ==============================


class HostBrain:
    def __init__(self):
        graph = StateGraph(HostState)

        graph.add_node("insights", extract_insights)
        graph.add_node("rate", rate_insights)

        graph.set_entry_point("insights")
        graph.add_edge("insights", "rate")
        graph.add_edge("rate", END)

        self.graph = graph.compile()

    async def run(self, essay: str):
        return await self.graph.ainvoke({"essay": essay})


# ==============================
# Host Orchestrator (PIPELINE)
# ==============================


class HostAgent:
    def __init__(self):
        self.client = A2AHostClient()
        self.brain = HostBrain()

    async def run(self, topic: str):
        print("\n🔗 Connecting to all friend agents...\n")

        friend1 = await self.client.connect("http://localhost:10001")
        friend2 = await self.client.connect("http://localhost:10002")
        friend3 = await self.client.connect("http://localhost:10003")

        # ---------------------------
        # STEP 1 → FRIEND 1 (RESEARCH)
        # ---------------------------
        print("📨 Sending TOPIC to Friend 1 (Research Agent)\n")
        research = await self.client.send(friend1, topic)

        print("========== 🔍 RESEARCH OUTPUT ==========\n")
        print(research)
        print("\n========================================\n")

        # ---------------------------
        # STEP 2 → FRIEND 2 (OUTLINES)
        # ---------------------------
        print("📨 Sending RESEARCH to Friend 2 (Outline Agent)\n")
        outlines = await self.client.send(friend2, research)

        print("========== 🧩 OUTLINES OUTPUT ==========\n")
        print(outlines)
        print("\n========================================\n")

        # ---------------------------
        # STEP 3 → FRIEND 3 (ESSAY)
        # ---------------------------
        print("📨 Sending OUTLINES to Friend 3 (Essay Agent)\n")
        essay = await self.client.send(friend3, outlines)

        print("========== ✍️ ESSAY OUTPUT ==========\n")
        print(essay)
        print("\n======================================\n")

        # ---------------------------
        # STEP 4 → HOST ANALYSIS
        # ---------------------------
        print("🧠 Host analyzing ESSAY...\n")
        result = await self.brain.run(essay)

        print("========== 💡 INSIGHTS ==========\n")
        print(result["insights"])
        print("\n===============================\n")

        print("========== ⭐ RATING ==========\n")
        print(f"{result['rating']} / 10")
        print("\n===============================\n")

        return result


# ==============================
# Program Entry
# ==============================


async def main():
    print("\n==============================")
    print("   A2A MULTI-AGENT SYSTEM   ")
    print("==============================\n")

    # ✅ USER INPUT (FLOW STARTS HERE)
    topic = input("Enter your topic: ")

    host = HostAgent()
    await host.run(topic)


if __name__ == "__main__":
    asyncio.run(main())
