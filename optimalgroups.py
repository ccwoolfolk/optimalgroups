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
optimized_value = prob.objective.value()
perfect_value = len(persons)
worst_value = sum([np.nanmax(x) for x in cost_matrix])
optimization_score = 100 - round(100 * (optimized_value - perfect_value) / (worst_value - perfect_value), 0)


# Display results
print(f"Optimization score (0-100): {optimization_score}")
print(f"Achieved: {optimized_value}")
print(f"Perfect: {perfect_value}")
print(f"Worst: {worst_value}")

for j in range(n_alternatives):
    selected_persons = [persons[i] for i in range(n_persons) if choices[i][j].varValue == 1.0]
    if len(selected_persons) > 0:
        print(f"\n{alternative_names[j]}:")
        [print(f"  {person}") for person in selected_persons]
