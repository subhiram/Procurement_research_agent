from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
# -------------------------
# 1. Define dummy tools
# -------------------------

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny and 28°C."


@tool
def calculator(expression: str) -> str:
    """Calculate a simple mathematical expression."""
    try:
        return str(eval(expression))
    except Exception:
        return "Invalid expression."


tools = [get_weather, calculator]


# -------------------------
# 2. Define the state
# -------------------------

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------------
# 3. Create the LLM
# -------------------------

from llm_router.router import LLMRouter


# llm = LLMRouter(
#     model="nemotron-super",
#     provider="nvidia_nim",
# )

llm = LLMRouter(
    model="gpt-oss-20b",
    provider="nvidia_nim",
)

llm_with_tools = llm.bind_tools(tools)

# -------------------------
# 4. Define the agent node
# -------------------------

def agent(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# -------------------------
# 5. Build the graph
# -------------------------

graph = StateGraph(State)

graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")

# If the LLM requests a tool -> tools
# Otherwise -> END
graph.add_conditional_edges(
    "agent",
    tools_condition,
)

graph.add_edge("tools", "agent")

app = graph.compile()


# -------------------------
# 6. Run the agent
# -------------------------

result = app.invoke({
    "messages": [
        ("user", "What's the weather in Hyderabad?")
    ]
})

print(type(result))
print(result["messages"])