"""

Reducers in LangGraph are used to tell LangGraph how to combine multiple updates to the same state key

"""

from langgraph.graph import StateGraph,START,END

from pydantic import BaseModel,Field

from typing import Annotated

import operator

from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(

    repo_id="meta-llama/Llama-3.1-8B-Instruct",

    task="text-generation"

)

chat_model = ChatHuggingFace(llm=llm)


class EvaluationSchema(BaseModel) :

    feedback : str = Field(description='Write the feedback of the essay')

    score : int = Field(description='Gimme the output in btn 0 to 10', ge=0, le=10)


structure_model = chat_model.with_structured_output(EvaluationSchema,    method = "json_schema")


class EssayState(BaseModel) :

    essay : str

    language_feedback : str

    analysis_feedback : str

    clarity_feedback : str

    overall_feedback : str

    individual_scores : Annotated[list, operator.add]

    avg_score : float



def evaluate_language(state : EssayState) -> dict :

    prompt = f"""

    Provide me an evaluation of the essay's language.

    Essay :

    {state.essay}

    """

    output = structure_model.invoke(prompt)

    return {
        'language_feedback' : output['feedback'],
        'individual_scores' : [output['score']]
    }



def analysis_feedback(state : EssayState) -> dict :

    prompt = f"""

    Analyse the essay and evaluate the quality of its analysis and arguments.

    Essay :

    {state.essay}

    """

    output = structure_model.invoke(prompt)
    print(output)
    return {
        'analysis_feedback' : output['feedback'],
        'individual_scores' : [output['score']]
    }



def evaluate_thought(state : EssayState) -> dict :

    prompt = f"""

    Evaluate the clarity of thought and ideas in the essay.

    Essay :

    {state.essay}

    """

    output = structure_model.invoke(prompt)

    return {
        'clarity_feedback' : output['feedback'],
        'individual_scores' : [output['score']]
    }


def final_evaluation(state : EssayState) -> dict:

    scores = state.individual_scores
    average_score = sum(scores) / len(scores) if scores else 0

    return {
        'overall_feedback': (
            f"Language: {state.language_feedback}\n"
            f"Analysis: {state.analysis_feedback}\n"
            f"Clarity: {state.clarity_feedback}"
        ),
        'avg_score': average_score
    }


graph = StateGraph(EssayState)


# nodes

graph.add_node('evaluate_language', evaluate_language)

graph.add_node('analysis_feedback', analysis_feedback)

graph.add_node('evaluate_thought', evaluate_thought)

graph.add_node('final_evaluation', final_evaluation)


# edges

graph.add_edge(START, 'evaluate_language')

graph.add_edge(START, 'analysis_feedback')

graph.add_edge(START, 'evaluate_thought')


graph.add_edge('evaluate_language', 'final_evaluation')

graph.add_edge('analysis_feedback', 'final_evaluation')

graph.add_edge('evaluate_thought', 'final_evaluation')


graph.add_edge('final_evaluation', END)


workflow = graph.compile()


initial_state = EssayState(

    essay = """

    Artificial intelligence is changing the world rapidly.
    It helps humans automate repetitive tasks and solve complex problems.
    However, AI should be used responsibly because improper use can
    create ethical and social problems.

    """,

    language_feedback = "",

    analysis_feedback = "",

    clarity_feedback = "",

    overall_feedback = "",

    individual_scores = [],

    avg_score = 0

)


final_state = workflow.invoke(initial_state)


print(final_state)

print(final_state['overall_feedback'])