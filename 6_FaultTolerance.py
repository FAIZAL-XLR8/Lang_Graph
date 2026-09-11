from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict
import time
#"to simulate fault tolerance just invoke with none as input in invoke call"

# 1. Define the state
class CrashState(TypedDict):
    input: str
    step1: str
    step2: str
# 2. Define steps
def step_1(state: CrashState) -> CrashState:
    print("✅ Step 1 executed")
    return {"step1": "done", "input": state["input"]}

def step_2(state: CrashState) -> CrashState:
    print("⏳ Step 2 hanging... now manually interrupt from the notebook toolbar (STOP button)")
    time.sleep(1000)  # Simulate long-running hang
    return {"step2": "done"}

def step_3(state: CrashState) -> CrashState:
    print("✅ Step 3 executed")
    return {"done": True}
# 3. Build the graph
builder = StateGraph(CrashState)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)

builder.set_entry_point("step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
try:
    print("▶️ Running graph: Please manually interrupt during Step 2...")
    graph.invoke({"input": "start"}, config={"configurable": {"thread_id": 'thread-1'}})
except KeyboardInterrupt:
    print("❌ Kernel manually interrupted (crash simulated).")
print("\n🔁 Re-running the graph to demonstrate fault tolerance...")
final_state = graph.invoke(None, config={"configurable": {"thread_id": 'thread-1'}})
print("\n✅ Final State:", final_state)
list(graph.get_state_history({"configurable": {"thread_id": 'thread-1'}}))
#output-->notice that step1 is not repeated here, it contniues from where it crashed and we send final state as None
# ▶️ Running graph: Please manually interrupt during Step 2...
# ✅ Step 1 executed
# ⏳ Step 2 hanging... now manually interrupt from the notebook toolbar (STOP button)
# ❌ Kernel manually interrupted (crash simulated).

# 🔁 Re-running the graph to demonstrate fault tolerance...
# ⏳ Step 2 hanging... now manually interrupt from the notebook toolbar (STOP button)
