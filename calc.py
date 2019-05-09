import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpMaximize, LpAffineExpression

rawdata = pd.read_excel('test-data.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
prob = LpProblem("Optimal 10x Grouping", LpMinimize)

cost_matrix = np.array([
    [2,3,1,10],
    [10,1,2,3],
])

(n_agents, n_alternatives) = cost_matrix.shape

costs = cost_matrix.flatten()

# Create binary Lp variables for every agent/alternative combination
choices = []
for i in range(n_agents):
    choices.append([])
    for j in range(n_alternatives):
        choices[i].append(LpVariable(f"Choice{i},{j}", cat='Binary'))

has_membership = [LpVariable(f"Choice{j}HasMembership", cat='Binary') for j in range(n_alternatives)]

# Add constraints
for i in range(n_agents):
    prob += sum(choices[i]) == 1 # Only one result per agent

min_in_alternative = 1 # i.e., min ppl in a group if > 0
for j in range(n_alternatives):
    prob += has_membership[j] <= 1
    for i in range(n_agents):
        prob += has_membership[j] >= choices[i][j]
    # If a group has any members, enforce a minimum number
    prob += sum([choices[i][j] for i in range(n_agents)]) >= min_in_alternative * has_membership[j]

choices_full = np.array(choices).flatten()
prob += lpSum([choices_full[i] * costs[i] for i in range(len(choices_full))])

prob.solve()
for v in prob.variables():
    if v.varValue>0:
        print(v.name, "=", v.varValue)
