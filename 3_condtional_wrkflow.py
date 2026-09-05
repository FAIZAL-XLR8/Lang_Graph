from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel
from typing import Annotated,Literal
import operator
#logic is to form one condtional_edge(from node1 to condition(which would return a name of the node2))
#then from each condtional candidate node make an edge to next step node
# yani graph.add_condtional_node(node1, condtional_func)
# condtional_func returns the name of the node accr to the condtion
# graph.add_node(candidate1, node 2)
# graph.add_node(candiate2, node2)
class EquationState (BaseModel) :
    a : int
    b : int
    c:int
    equation : str = ""
    discriminant : float = 0.0
    roots : list[str] = []
def print_equation(state : EquationState) -> dict :
    temp = f"{state.a}x2{state.b}x{state.c}"
    return {'equation' : temp}
def calculate_discriminant(state : EquationState) -> dict :
    ds = (state.b ** 2) - 4*(state.a)*(state.c)
    return {'discriminant' : ds}
def imaginary_roots(state : EquationState) -> dict :
    return {'roots' : ['imaginary_roots']}
def real_roots(state : EquationState) -> dict :
    return {'roots' : ['real_roots']}
def condition_check(state : EquationState) -> Literal['real_roots', 'imaginary_roots'] :
    if state.discriminant >= 0 : return 'real_roots'
    elif state.discriminant < 0 : return 'imaginary_roots'

graph = StateGraph(EquationState)
graph.add_node('print_equation', print_equation)  
graph.add_node('calculate_discriminant', calculate_discriminant)
graph.add_node('real_roots', real_roots)
graph.add_node('imaginary_roots', imaginary_roots)
graph.add_edge(START, 'print_equation')
graph.add_edge('print_equation', 'calculate_discriminant')
graph.add_conditional_edges('calculate_discriminant', condition_check, {'real_roots' : 'real_roots', 'imaginary_roots': 'imaginary_roots'})
graph.add_edge('imaginary_roots', END)
graph.add_edge('real_roots', END)
workflow = graph.compile()
initial_state = {'a' : 1, 'b' : 10, 'c' : 1}
result = workflow.invoke(initial_state)
print(result)