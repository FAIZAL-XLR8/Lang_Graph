from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel
class BMI_State(BaseModel) :
    wieght : float
    hieght : float
    result : float
    label_bmi : str 

#each node is just a pythin functiojn which expects a state object and return a state object so each function should fllow this rule
def calculate_bmi(state : BMI_State) ->BMI_State :
    wt = state.wieght
    ht = state.hieght
    state.result = round(wt / (ht**2), 2)
    return state
def label_bmi(state: BMI_State) -> BMI_State :
        temp = ""
        if state.result > 18.5 and state.result < 25:
             temp = 'normal'
        elif state.result > 20:
            temp = 'overwieght'
        state.label_bmi = temp
        return state



graph = StateGraph(BMI_State)
#add nodes
graph.add_node('calculate_bmi', calculate_bmi)
graph.add_node('label_bmi', label_bmi)
#add edges
#start and end are dummy nodes
graph.add_edge(START, 'calculate_bmi')
graph.add_edge('calculate_bmi', 'label_bmi')
graph.add_edge('label_bmi', END)
#compile graph
workflow = graph.compile()
#excute graph
intial_state = {'wieght' : 64, 'hieght' : 1.7, 'result' : 0.0, 'label_bmi' : ''}
final_state = workflow.invoke(intial_state)
print(final_state)