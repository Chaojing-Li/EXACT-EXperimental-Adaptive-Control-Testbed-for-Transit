from typing import Dict, List, Tuple
from tabulate import tabulate


def find_terminal_to_common_routes(route_terminal: Dict[str, str]) -> Dict[str, List[str]]:
    terminal_to_routes = {}
    for route, terminal in route_terminal.items():
        if terminal not in terminal_to_routes:
            terminal_to_routes[terminal] = []
        terminal_to_routes[terminal].append(route)
    return terminal_to_routes


def print_od_table(od_table: Dict[str, Dict[str, float]], visited_stop_id: List[str]) -> None:
    """
    Prints a dictionary of dictionaries as a matrix-like table, following the sequence of visited stop IDs.

    Args:
        od_table (Dict[str, Dict[str, float]]): The dictionary of dictionaries representing the matrix.
                                                 The outer keys represent the row names, and the inner keys represent the column names.
                                                 The inner values represent the matrix entries.
        visited_stop_id (List[str]): The list of visited stop IDs representing the sequence of rows and columns in the matrix.

    Returns:
        None
    """
    # Create a list of rows with the row name and corresponding values, following the sequence of visited stop IDs
    rows = []
    for stop_id in visited_stop_id:
        if stop_id in od_table:
            row = [
                stop_id] + [od_table[stop_id].get(col_id, 0.0) for col_id in visited_stop_id]
            rows.append(row)

    # Print the matrix using tabulate
    print(tabulate(rows, headers=[""] + visited_stop_id,
          floatfmt=".4f", tablefmt="compressed"))


def sum_entries(od_table: Dict[str, Dict[str, float]]) -> float:
    """
    Calculates the sum of all entries' values in a dictionary of dictionaries.

    Args:
        od_table (Dict[str, Dict[str, float]]): The dictionary of dictionaries representing the matrix.
                                                 The outer keys represent the row names, and the inner keys represent the column names.
                                                 The inner values represent the matrix entries.

    Returns:
        float: The sum of all entries' values in the dictionary.
    """
    total_sum = 0.0

    for inner_dict in od_table.values():
        for value in inner_dict.values():
            total_sum += value

    return total_sum


def sum_entries_with_row_sums(od_table: Dict[str, Dict[str, float]]) -> Tuple[float, Dict[str, float]]:
    """
    Calculates the sum of all entries' values and the sum of each row in a dictionary of dictionaries.

    Args:
        od_table (Dict[str, Dict[str, float]]): The dictionary of dictionaries representing the matrix.
                                                 The outer keys represent the row names, and the inner keys represent the column names.
                                                 The inner values represent the matrix entries.

    Returns:
        Tuple[float, Dict[str, float]]: A tuple containing the sum of all entries' values and a dictionary of row sums.
                                        The dictionary of row sums has the row names as keys and the corresponding row sums as values.
    """
    total_sum = 0.0
    row_sums = {}

    for row_name, inner_dict in od_table.items():
        row_sum = sum(inner_dict.values())
        row_sums[row_name] = row_sum
        total_sum += row_sum

    return total_sum, row_sums


if __name__ == '__main__':
    route_terminal = {'0': '0', '1': '1', '2': '0'}
    terminal_to_routes = find_terminal_to_common_routes(route_terminal)
