from langchain_core.messages import HumanMessage,SystemMessage,BaseMessage
from langgraph.graph import StateGraph,START,END
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from typing import TypedDict, Annotated
from langgraph.graph import add_messages #this is an operator like to add a message
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith import traceable
import sqlite3
load_dotenv()
chat_model = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
       repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        max_new_tokens=100,
    ) 
)

conn = sqlite3.connect(database='chat_bot.db',check_same_thread=False)
checkpoint = SqliteSaver(conn=conn)
class chatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

@traceable(name="chat_node", tags=['chat_model'],metadata={'dimension' : 'chat_app'})
def chat_node(state : chatState) :
    new_message = chat_model.invoke(state['messages'])
    return {'messages' : [new_message]}



graph = StateGraph(chatState)
graph.add_node('chat_node', chat_node)
#edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)
workflow = graph.compile(checkpointer=checkpoint)
# initial_state = {
#     'messages' : [HumanMessage('What is the capital of India')]
# }


configuration = {
    "configurable" : {"thread_id" : "3"},
    "metadata" : {
        "thread_id" : 3,
        "user" : "Faizal"
    },
    "run_name" : "chat_bot"
    #mentioning thread_id in configurable does 2 things
    #1) it will save the run in the sqlite database with the thread_id
    #2) it will save the run in the langsmith with the thread_id
    #the metadata also saved in langsmith with each corresponding run's thread
    #mentioning run_name in config, it will save the run in the langsmith with the run_name
}
while True :
    user_input = input("tell me how may I help?")
    print(user_input)
    if user_input.strip().lower() in ['exit', 'bye', 'end']:
        break
    initial_state = {
    'messages' : [HumanMessage(content=user_input)]
    }

    # workflow.invoke(initial_state)
    # res = workflow.invoke(initial_state)
    # print(res['messages'][-1].content)
    for message_chunk, metadata in workflow.stream(
    initial_state,
     config=configuration,
    stream_mode="messages"
    ):
        if message_chunk.content:
            print(message_chunk.content, end="", flush=True)
    print()  # once, after the loop, not inside it
# for item in checkpoint.list(None):
#     print(item)