"""
Reducers in LangGraph are used to tell LangGraph how to combine multiple updates to the same state key
"""
from langgraph.graph import StateGraph
from pydantic import BaseModel,Field
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
chat_model = ChatHuggingFace(llm=llm)
class EvaluationSchema(BaseModel) :
    feedback : str = Field(description='Write the feedback of the essay'),
    score: int = Field(description='Gimme the output in btn 0 to 10', ge=0, le=10)
structure_model = chat_model.with_structured_output(EvaluationSchema)
