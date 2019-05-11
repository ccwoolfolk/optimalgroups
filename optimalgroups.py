import argparse
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum

NAN_VALUE = 10000 # Arbitrarily high value to force optimization away from empty choices


parser = argparse.ArgumentParser(description='Sort a group into optimal preference-based subgroups.')
parser.add_argument('min', type=int, help='Min group size if group has > 0')
parser.add_argument('file_path', help='Path to input file with "Persons" in first column')

args = parser.parse_args()


# Assemble data
rawdata = pd.read_excel(args.file_path, 'Sheet1', index_col=None, na_values=['NA'])
prob = LpProblem("Optimal 10x Grouping", LpMinimize)

alternative_names = list(rawdata.columns[1:])
persons = list(rawdata['Persons'])
cost_matrix = rawdata.drop('Persons', 'columns').to_numpy()
(n_persons, n_alternatives) = cost_matrix.shape

costs = [NAN_VALUE if np.isnan(x) else x for x in cost_matrix.flatten()]


# Create binary LpVariables for every agent/alternative combination
choices = []
for i in range(n_persons):
    choices.append([LpVariable(f"Choice:{persons[i]},{alternative_names[j]}", cat='Binary') for j in range(n_alternatives)])


# Create binary LpVariables tracking if a group has any members
has_membership = [LpVariable(f"{alternative_names[j]}:HasMembership", cat='Binary') for j in range(n_alternatives)]


# Add constraints
for i in range(n_persons):
    prob += sum(choices[i]) == 1 # Only one result per agent

for j in range(n_alternatives):
    prob += has_membership[j] <= 1
    for i in range(n_persons):
        prob += has_membership[j] >= choices[i][j]
    # If a group has any members, enforce a minimum number
    prob += sum([choices[i][j] for i in range(n_persons)]) >= args.min * has_membership[j]


# Define and calculate the optimization
choices_full = np.array(choices).flatten()
prob += lpSum([choices_full[i] * costs[i] for i in range(len(choices_full))])
prob.solve()


# Display results
print(prob)
for v in prob.variables():
    if v.varValue>0:
        print(v.name, "=", v.varValue)
