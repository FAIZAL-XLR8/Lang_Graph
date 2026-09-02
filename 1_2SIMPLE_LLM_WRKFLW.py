from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langgraph.graph import StateGraph
from pydantic import BaseModel
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-3B-Instruct',
    task='text-generation'
)
chat_model = ChatHuggingFace(llm=llm)
class LLMState(BaseModel):
    ques : str
    ans : str


def llm_qa(state : LLMState) ->LLMState :
    #extract the ques first
    ques = state.ques
    #make a prompt
    prompt = f"answer the following ques {ques}"
    #llm call
    res = chat_model.invoke(prompt).content
    state.ans = res
    return state
#create a graph
graph = StateGraph(LLMState)
#add nodes
graph.add_node('llm_qa', llm_qa)
