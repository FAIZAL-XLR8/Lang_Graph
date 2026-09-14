from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.tools import tool
import sqlite3
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langsmith import traceable
import os

os.environ["LANGCHAIN_PROJECT"] = "tool_call"

load_dotenv()

class chatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # In LangGraph, Annotated is used to define a reducer 
    # function for a state channel.

sqlite_db_connection = sqlite3.connect("new_chat_bot_db.db", check_same_thread=False)
checkpoint = SqliteSaver(conn=sqlite_db_connection)

@tool
def calculator(a: float, b: float, operation: str) -> dict:
    """
    Perform basic arithmetic operations: add, subtract, divide, multiply.
    IMPORTANT: Use this tool ONLY if the user explicitly asks to calculate something.
    Do NOT call this tool for dates, years, or numbers mentioned in factual search results.
    """
    if operation == "add":
        return {"a": a, "b": b, "operation": operation, "result": a + b}
    elif operation == "subtract":
        return {"a": a, "b": b, "operation": operation, "result": a - b}
    elif operation == "divide":
        if b == 0:
            return {"a": a, "b": b, "operation": operation, "result": "error_cant_divide_with_zero"}
        else:
            return {"a": a, "b": b, "operation": operation, "result": a / b}
    elif operation == "multiply":
        return {"a": a, "b": b, "operation": operation, "result": a * b}
    else:
        return {"a": a, "b": b, "operation": operation, "result": "no_such_operator_exists"}

# Tools definition
wrapper = DuckDuckGoSearchAPIWrapper(region="en-in")
search_engine_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
tools = [calculator, search_engine_tool]
tools_node = ToolNode(tools)

# Model and LLM with tools binding
chat_model = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        max_new_tokens=500,
    )
)
llm_with_tools = chat_model.bind_tools(tools)

system_prompt = SystemMessage(
    content=(
        "You are a helpful assistant with access to tools:\n"
        "1. duckduckgo_search: Use this to search for current events and factual information.\n"
        "2. calculator: Use this ONLY if the user explicitly asks for mathematical calculations.\n"
        "Never use the calculator on dates, years, terms, or numbers extracted from search results.\n"
        "Once you have gathered sufficient information to answer the question, formulate your answer directly."
    )
)

# Chat node
@traceable(name="chat_llm", tags=["llm"])
def chat_node(state: chatState) -> dict:
    messages = state["messages"]
    # Ensure system prompt is present to instruct the LLM on tool usage
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [system_prompt] + list(messages)
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# Graph construction
graph = StateGraph(chatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools_node", tools_node)

# Edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition, {
    "tools": "tools_node",
    END: END
})
graph.add_edge("tools_node", "chat_node")

# Compile graph with checkpointer
workflow = graph.compile(checkpointer=checkpoint)

config = {
    "configurable": {"thread_id": "3"},
    "metadata": {"tag": "bind_tools_lec"},
    "run_name": "bind_tools_lec"
}

response = workflow.invoke(
    {"messages": [HumanMessage(content="who is the current prime minister of india?")]},
    config=config
)

print(response["messages"][-1].content)

