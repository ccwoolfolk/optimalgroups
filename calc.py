import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum

MIN_IN_ALTERNATIVE = 4 # Minimum number of people (excl. 0) that can constitute a group
NAN_VALUE = 10000 # Arbitrarily high value to force optimization away from empty choices
rawdata = pd.read_excel('test-data.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
prob = LpProblem("Optimal 10x Grouping", LpMinimize)

alternative_names = list(rawdata.columns[1:])
persons = list(rawdata['Persons'])
cost_matrix = rawdata.drop('Persons', 'columns').to_numpy()

(n_persons, n_alternatives) = cost_matrix.shape

costs = [NAN_VALUE if np.isnan(x) else x for x in cost_matrix.flatten()]

# Binary LpVariables for every agent/alternative combination
choices = []
for i in range(n_persons):
    choices.append([LpVariable(f"Choice:{persons[i]},{alternative_names[j]}", cat='Binary') for j in range(n_alternatives)])

# Binary LpVariables tracking if a group has any members
has_membership = [LpVariable(f"{alternative_names[j]}:HasMembership", cat='Binary') for j in range(n_alternatives)]

# Add constraints
for i in range(n_persons):
    prob += sum(choices[i]) == 1 # Only one result per agent

for j in range(n_alternatives):
    prob += has_membership[j] <= 1
    for i in range(n_persons):
        prob += has_membership[j] >= choices[i][j]
    # If a group has any members, enforce a minimum number
    prob += sum([choices[i][j] for i in range(n_persons)]) >= MIN_IN_ALTERNATIVE * has_membership[j]

choices_full = np.array(choices).flatten()
prob += lpSum([choices_full[i] * costs[i] for i in range(len(choices_full))])

prob.solve()
for v in prob.variables():
    if v.varValue>0:
        print(v.name, "=", v.varValue)
