from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel
class BMI_State(BaseModel) :
    wieght : float
    hieght : float
    result : float

#each node is just a pythin functiojn which expects a state object and return a state object so each function should fllow this rule
def calculate_bmi(state : BMI_State) ->BMI_State :
    wt = state.wieght
    ht = state.hieght
    state.result = round(wt / (ht**2), 2)
    return state
graph = StateGraph(BMI_State)
#add nodes
graph.add_node('calculate_bmi', calculate_bmi)
#add edges
#start and end are dummy nodes
graph.add_edge(START, 'calculate_bmi')
graph.add_edge('calculate_bmi', END)
#compile graph
workflow = graph.compile()
#excute graph
intial_state = {'wieght' : 64, 'hieght' : 1.7, 'result' : 0.0}
final_state = workflow.invoke(intial_state)
print(final_state)