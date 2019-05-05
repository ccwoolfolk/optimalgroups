import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpMaximize, LpAffineExpression

rawdata = pd.read_excel('test-data.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
prob = LpProblem("Optimal 10x Grouping", LpMinimize)

cost_matrix = [
    [2,3,1,10],
    [10,2,1,3],
]

costs = cost_matrix[0] + cost_matrix[1]

choices = []
for i in range(2):
    choices.append([])
    for j in range(4):
        choices[i].append(LpVariable(f"Choice{i},{j}", cat='Binary'))
    prob += choices[i][0] + choices[i][1] + choices[i][2] + choices[i][3] == 1


choices_full = choices[0] + choices[1]
tmp = LpAffineExpression([(choices_full[i], costs[i]) for i in range(len(choices_full))])
prob += lpSum([choices_full[i] * costs[i] for i in range(len(choices_full))])
# prob += lpSum(tmp)
print(choices_full)
print(costs)

prob.solve()
print(choices_full[0] * costs[0])
for v in prob.variables():
    if v.varValue>0:
        print(v.name, "=", v.varValue)
print(prob.constraints)
