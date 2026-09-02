from langgraph.graph import StateGraph,START,END

from pydantic import BaseModel

class CricketState(BaseModel):

    runs : int

    balls : int

    sixes : int

    fours : int

    strike_rate : float

    balls_per_boundary : float

    boundary_percent : float

    summary : str


def strike_rate(state : CricketState) -> dict :

    temp = (state.runs / state.balls * 100)

    return {'strike_rate': temp}


def boundary_percent(state : CricketState) -> dict :

    temp = ((state.fours * 4 + state.sixes * 6) / state.runs) * 100

    return {'boundary_percent': temp}


def balls_per_boundary(state : CricketState) -> dict :

    total_boundary = state.fours + state.sixes

    temp = state.balls / total_boundary

    return {'balls_per_boundary': temp}


def summary(state : CricketState) -> dict :

    temp = f"""Strike_rate is {state.strike_rate}

balls_per_boundary is {state.balls_per_boundary}

boundary_percent is {state.boundary_percent}
"""

    return {'summary':temp}


graph = StateGraph(CricketState)


# nodes

graph.add_node('strike_rate', strike_rate)

graph.add_node('boundary_percent', boundary_percent)

graph.add_node('balls_per_boundary', balls_per_boundary)

graph.add_node('summary', summary)


# edges

graph.add_edge(START, 'balls_per_boundary')

graph.add_edge(START, 'boundary_percent')

graph.add_edge(START, 'strike_rate')

graph.add_edge('strike_rate', 'summary')

graph.add_edge('boundary_percent', 'summary')

graph.add_edge('balls_per_boundary', 'summary')

graph.add_edge('summary', END)


workflow = graph.compile()


initial_state = CricketState(
    runs = 120,
    balls = 80,
    sixes = 6,
    fours = 10,
    strike_rate = 0,
    balls_per_boundary = 0,
    boundary_percent = 0,
    summary = ""
)


final_state = workflow.invoke(initial_state)


print(final_state)

print(final_state['summary'])