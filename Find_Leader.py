# Importing the numpy library, which provides support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
import numpy as np

# Importing the pandas library, which provides data manipulation and analysis capabilities.
import pandas as pd

# Importing the random module, which can generate random numbers.
import random

# Importing the warnings module, which allows us to manage warning messages.
import warnings

# Using the 'filterwarnings' function from the 'warnings' module to ignore all warning messages.(for clean terminal)
warnings.filterwarnings("ignore")


# Define a function named 'load_and_preprocess_data' that takes a filename as an argument and return clean dataframe and original dataframe
def load_and_preprocess_data(filename):
    # Use pandas to read the CSV file specified by the filename. Set the index column to be the second column (index 1). Drop the "Timestamp" column from the dataframe.
    data = pd.read_csv(filename, index_col=1).drop("Timestamp", axis=1)
    # Create a copy of the data dataframe and store it in 'original_data'. This is done to keep the original data intact for future use.
    original_data = data.copy()

    # Start preprocessing the 'data' dataframe.
    # Apply a lambda function to each element in the dataframe. The lambda function converts each element to a string, takes the last 11 characters, and converts them to uppercase.
    data = data.applymap(lambda x: str(x)[-11:].upper())
    # Rename the columns of the dataframe. The new column names are the old names with the first 5 characters removed.
    data.columns = data.columns.map(lambda x: x[5:])
    # Rename the index of the dataframe. The new index names are the old names with the first 11 characters converted to uppercase.
    data.index = data.index.map(lambda x : x[0:11].upper())
    # Rename the index of the dataframe to "Entry Number".
    data.index.name = "Entry Number"
    # Return the preprocessed 'data' dataframe and the 'original_data' dataframe.
    return data, original_data

# Define a function named 'extract_students' that takes a dataframe 'original_data' as an argument.
def extract_students(original_data):
    # Flatten the 'original_data' dataframe into a 1D array, convert it to a list, and then convert it to a set to remove duplicates. Store the result in 'student_entries'.
    student_entries = set(original_data.values.flatten().tolist())
    # Remove any NaN values from the 'student_entries' set.
    student_entries.remove(np.nan)
    # Create an empty pandas Series with datatype string to store the student names. The index of this series will be the last 11 characters of each entry, and the value will be the rest of the entry.
    student_names = pd.Series(dtype=str)
    # Iterate over each entry in 'student_entries'.
    for entry in student_entries:
        # For each entry, use the last 11 characters as the index for 'student_names', and the rest of the entry as the value.
        student_names.loc[entry.upper()[-11:]] = entry.upper()[0:-11]
    # Return the 'student_names' series.
    return student_names

# Define a function named 'apply_student_names' that takes a dataframe 'data' and a series 'student_names' as arguments.
def apply_student_names(data, student_names):
    # Apply a lambda function to each element in the 'data' dataframe. The lambda function replaces each element with the corresponding student name from 'student_names'. If the element does not exist in 'student_names', it is replaced with "NAN".
    data = data.applymap(lambda x: student_names.get(x, "NAN"))
    # Apply a lambda function to each index in the 'data' dataframe. The lambda function replaces each index with the corresponding student name from 'student_names'. If the index does not exist in 'student_names', it is replaced with "NAN".
    data.index = data.index.map(lambda x: student_names.get(x, "NAN"))
    
    # Return the modified 'data' dataframe.
    return data

# Define a function named 'perform_random_walk' that takes a dataframe 'data' as an argument.
def perform_random_walk(data):
    # Initialize an empty dictionary to store the visit counts for each node.
    visit_counts = {}
    # Randomly select a node from the index of the 'data' dataframe to start the random walk.
    node = random.choice(data.index)
    # Perform the random walk 10^6 times.
    for _ in range(10**6):
        # If the selected node is not in the index of the 'data' dataframe, select a new node randomly.
        while node not in data.index:
            # Increment the visit count for the current node by 1/10^6. If the node is not already in 'visit_counts', it is added with a default count of 0 before incrementing.
            visit_counts[node] = visit_counts.get(node,0) + 1/10**6
            #choose the node
            node = random.choice(data.index)
        # Increment the visit count for the current node by 1/10^6. If the node is not already in 'visit_counts', it is added with a default count of 0 before incrementing.
        visit_counts[node] = visit_counts.get(node,0) + 1/10**6
        # Randomly select the next node from the values in the row corresponding to the current node in the 'data' dataframe.
        next_node = random.choice(data.loc[node].values)
        # If the selected next node is "NAN", select a new next node randomly from the same row.
        while next_node == "NAN":
            #choose the node
            next_node = random.choice(data.loc[node])
        # Set the current node to the selected next node for the next iteration.
        node = next_node

    # Return the 'visit_counts' dictionary.
    return visit_counts


# Define a function named 'main' that doesn't take any arguments.
def main():
    # Call the 'load_and_preprocess_data' function with the filename "Impression_Network.csv" and store the returned dataframes in 'data' and 'original_data'.
    data, original_data = load_and_preprocess_data("Impression_Network.csv")
    # Call the 'extract_students' function with the 'original_data' dataframe and store the returned series in 'student_names'.
    student_names = extract_students(original_data)
    # Call the 'perform_random_walk' function with the 'data' dataframe and store the returned dictionary in 'visit_counts'.
    visit_counts = perform_random_walk(data)
    # Sort the keys of the 'visit_counts' dictionary in descending order of their values and store the sorted keys in 'sorted_nodes'.
    sorted_nodes = sorted(visit_counts, key=visit_counts.get, reverse=True)
    # Print the student name corresponding to the first node in 'sorted_nodes'.
    print(student_names[sorted_nodes[0]],sorted_nodes[0])

# Check if the script is being run directly (not being imported). If it is, call the 'main' function.
if __name__ == "__main__":
    #use main() function
    main()
