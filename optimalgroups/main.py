"""CLI for optimal subgroups based on individual preferences"""
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum

NAN_VALUE = 10000 # Arbitrarily high value to force optimization away from empty choices


def optimize():
    """Run the optimizer"""
    args = get_args()
    rawdata = pd.read_excel(args.file_path, 'Sheet1', index_col=None, na_values=['NA'])
    prob = LpProblem("Optimal 10x Grouping", LpMinimize)

    alternatives = list(rawdata.columns[1:]) # Column labels
    persons = list(rawdata['Persons'])
    cost_matrix = rawdata.drop('Persons', 'columns').to_numpy() # Numpy array: [person][class]
    (n_persons, n_alternatives) = cost_matrix.shape

    costs = [NAN_VALUE if np.isnan(x) else x for x in cost_matrix.flatten()]


    # Create binary LpVariables for every person/alternative combination
    choices = []
    make_name = make_var_name_factory(persons, alternatives)
    for i in range(n_persons):
        choices.append([LpVariable(make_name(i, j), cat='Binary') for j in range(n_alternatives)])


    # Create binary LpVariables tracking if a group has any members
    has_membership = [LpVariable(f"{alternatives[j]}", cat='Binary') for j in range(n_alternatives)]


    # Add constraints
    # See https://cs.stackexchange.com/questions/12102/express-boolean-logic-operations-in-zero-one-
    #     integer-linear-programming-ilp
    # for a good explanation of logical operations (AND, OR, etc.) via linear constraints
    for i in range(n_persons):
        prob += sum(choices[i]) == 1 # Only one result per person

    for j in range(n_alternatives):
        # If the sum of every choice for the alternative is 0, has_membership must be 0
        prob += has_membership[j] <= sum([choices[i][j] for i in range(n_persons)])

        for i in range(n_persons):
            prob += has_membership[j] >= choices[i][j] # has_membership is 1 if any choice is 1

        # If a group has any members, enforce a minimum number
        prob += sum([choices[i][j] for i in range(n_persons)]) >= args.min * has_membership[j]


    # Define and calculate the optimization
    choices_full = np.array(choices).flatten()
    prob += lpSum([choices_full[i] * costs[i] for i in range(len(choices_full))])
    prob.solve()

    display_results(
        optimized_value=prob.objective.value(),
        persons=persons,
        alternatives=alternatives,
        choices=choices,
        cost_matrix=cost_matrix,
    )


def get_args():
    """Parse CLI arguments"""
    parser = ArgumentParser(description='Sort a group into optimal preference-based subgroups.')
    parser.add_argument('min', type=int, help='Min group size if group has > 0')
    parser.add_argument('file_path', help='Path to input file with "Persons" in first column')

    return parser.parse_args()


def make_var_name_factory(person_list, alternative_list):
    """Create an LpVariable name factory that accepts indexes"""
    def name_factory(i, j):
        person = person_list[i]
        alternative = alternative_list[j]
        return f"Choice:{person},{alternative}"
    return name_factory


def display_results(optimized_value, persons, alternatives, choices, cost_matrix):
    """Print results to the console"""
    (n_persons, n_alternatives) = cost_matrix.shape
    perfect_value = n_persons
    worst_value = sum([np.nanmax(x) for x in cost_matrix])
    score = 100 - round(100 * (optimized_value - perfect_value) / (worst_value - perfect_value), 0)

    print(f"Optimization score (0-100): {score}")
    print(f"Achieved: {optimized_value}")
    print(f"Perfect: {perfect_value}")
    print(f"Worst: {worst_value}")

    for j in range(n_alternatives):
        selected = [persons[i] for i in range(n_persons) if choices[i][j].varValue == 1.0]
        if selected:
            print(f"\n{alternatives[j]}:")
            for person in selected:
                print(f"  {person}")


if __name__ == "__main__":
    optimize()
