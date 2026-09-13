from langchain_core.messages import HumanMessage,SystemMessage,BaseMessage
from langgraph.graph import StateGraph,START,END
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from typing import TypedDict, Annotated
from langgraph.graph import add_messages #this is an operator like to add a message
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
load_dotenv()
chat_model = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
       repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        max_new_tokens=100,
    )
)
checkpoint = InMemorySaver()
class chatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

def chat_node(state : chatState) :
    new_message = chat_model.invoke(state['messages'])
    return {'messages' : [new_message]}



graph = StateGraph(chatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)
workflow = graph.compile(checkpointer=checkpoint)
# initial_state = {
#     'messages' : [HumanMessage('What is the capital of India')]
# }
configuration = {
    "configurable" : {"thread_id" : "1"}
}
while True :
    user_input = input("tell me how may I help?")
    print(user_input)
    if user_input.strip().lower() in ['exit', 'bye', 'end']:
        break
    initial_state = {
    'messages' : [HumanMessage(content=user_input)]
    }
# }
#     workflow.invoke(initial_state)
#     res = workflow.invoke(initial_state)
#     print(res['messages'][-1].content)
    for message_chunk, metadata in workflow.stream(
        initial_state,
        config=configuration,
        stream_mode="messages"
    ):
        if message_chunk.content:
            print(message_chunk.content, end="", flush=True)

    print()  # print newline after each assistant response