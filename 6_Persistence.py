from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
load_dotenv()

llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
       repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        max_new_tokens=200,
    )
)
class JokeState(TypedDict):

    topic: str
    joke: str
    explanation: str
def generate_joke(state: JokeState):

    prompt = f'generate a joke on the topic {state["topic"]}'
    response = llm.invoke(prompt).content

    return {'joke': response}
def generate_explanation(state: JokeState):

    prompt = f'write an explanation for the joke - {state["joke"]}'
    response = llm.invoke(prompt).content

    return {'explanation': response}
graph = StateGraph(JokeState)

graph.add_node('generate_joke', generate_joke)
graph.add_node('generate_explanation', generate_explanation)

graph.add_edge(START, 'generate_joke')
graph.add_edge('generate_joke', 'generate_explanation')
graph.add_edge('generate_explanation', END)

checkpointer = InMemorySaver()

workflow = graph.compile(checkpointer=checkpointer)
config1 = {"configurable": {"thread_id": "1"}}
config2 = {"configurable" : {"thread_id" : "2"}}
result = workflow.invoke({'topic':'pizza'}, config=config1)
r = workflow.invoke({"topic" : "Madara Uchiha"}, config = config2)
#print(result)
#print(r)
print(workflow.get_state(config1)) #gets the latest state of the workflow
lst = list(workflow.get_state_history(config1)) #gets the all the checkpoints made during the traversal 
#print(lst)

# from langgraph.checkpoint.memory import InMemorySaver
# checkpointer = InMemorySaver()
# config1 = {"configurable": {"thread_id": "1"}}
# workflow = graph.compile(checkpointer=checkpointer)
# workflow.invoke({"topic" : "pizza", config=config1})