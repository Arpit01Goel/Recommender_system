# Importing the numpy library, which provides support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
import numpy as np

# Importing the pandas library, which provides data manipulation and analysis capabilities.
import pandas as pd

# Importing the sys module, which provides access to some variables used or maintained by the Python interpreter and to functions that interact strongly with the interpreter.
import sys

# Importing the warnings module, which allows us to manage warning messages.
import warnings

# Importing the LinearRegression class from the sklearn.linear_model module, which is a module in Scikit-learn. This class is used to perform linear regression.
from sklearn.linear_model import LinearRegression

# Use the 'filterwarnings' function from the 'warnings' module to ignore all warning messages.
warnings.filterwarnings("ignore")

# Define a function named 'load_and_preprocess_data' that takes a filename as an argument.
def load_and_preprocess_data(filename):
    # Load the data from the CSV file specified by 'filename' into a pandas dataframe. Set the index of the dataframe to be the second column (index 1) of the CSV file and drop the "Timestamp" column. Store the dataframe in 'data'.
    data = pd.read_csv(filename, index_col=1).drop("Timestamp", axis=1)
    # Make a copy of the 'data' dataframe and store it in 'original_data'.
    original_data = data.copy()
    # Preprocess the 'data' dataframe by applying a lambda function to each element. The lambda function converts the element to a string, takes the last 11 characters, and converts them to uppercase.
    data = data.applymap(lambda x: str(x)[-11:].upper())
    # Replace the column names of the 'data' dataframe with the substring of each column name starting from the 6th character.
    data.columns = data.columns.map(lambda x: x[5:])
    # Replace the index of the 'data' dataframe with the substring of each index up to the 11th character, converted to uppercase.
    data.index = data.index.map(lambda x : x[0:11].upper())
    # Set the name of the index of the 'data' dataframe to "Entry Number".
    data.index.name = "Entry Number"
    # Return the preprocessed 'data' dataframe and the original 'data' dataframe.
    return data, original_data

# Define a function named 'extract_students' that takes a dataframe 'original_data' as an argument.
def extract_students(original_data):
    # Flatten the 'original_data' dataframe into a 1D array, convert it to a list, and then convert it to a set to remove duplicates. Store the set in 'student_entries'.
    student_entries = set(original_data.values.flatten().tolist())
    # Remove the numpy NaN value from 'student_entries'.
    student_entries.remove(np.nan)
    # Initialize an empty pandas Series with dtype string and store it in 'student_names'.
    student_names = pd.Series(dtype=str)
    # For each entry in 'student_entries', add an element to 'student_names' with the key as the last 11 characters of the entry (converted to uppercase) and the value as all characters of the entry except the last 11 (converted to uppercase).
    for entry in student_entries:
        student_names.loc[entry.upper()[-11:]] = entry.upper()[0:-11]
    # Return the 'student_names' Series.
    return student_names

# Define a function named 'apply_student_names' that takes a dataframe 'data' and a Series 'student_names' as arguments.
def apply_student_names(data, student_names):
    # Replace each element in 'data' with the corresponding value in 'student_names'. If the element is not a key in 'student_names', replace it with "NAN".
    data = data.applymap(lambda x: student_names.get(x, "NAN")+" "+x)
    # Replace each index in 'data' with the corresponding value in 'student_names'. If the index is not a key in 'student_names', replace it with "NAN".
    data.index = data.index.map(lambda x: student_names.get(x, "NAN")+" " + x)
    # Return the modified 'data' dataframe.
    return data

# Define a function named 'create_matrix' that takes a dataframe 'data' as an argument.
def create_matrix(data):
    # Flatten the 'data' dataframe into a 1D array, convert it to a list, and then convert it to a set to remove duplicates. Convert the set back to a list and store it in 'students'.
    students = list(set(data.values.flatten()))
    # Remove the string "NAN" from 'students'.
    students.remove("NAN")
    # Initialize a pandas dataframe with 'students' as both the index and columns, and all elements set to 0. Store the dataframe in 'matrix'.
    matrix = pd.DataFrame(index=students, columns=students)
    matrix[:] = 0
    # For each index in 'data', get the unique values in the corresponding row, and set the corresponding elements in 'matrix' to 1.
    for i in data.index:
        unique_columns = pd.Series(data.loc[i].values).drop_duplicates().values
        matrix.loc[i, unique_columns] = 1

    # Drop the column "NAN" from 'matrix'.
    matrix.drop("NAN", axis=1, inplace=True)
    
    # Return the 'matrix' dataframe.
    return matrix

# Call the 'load_and_preprocess_data' function with the filename "Impression_Network.csv" and store the returned dataframes in 'data' and 'original_data'.
data, original_data = load_and_preprocess_data("Impression_Network.csv")

# Call the 'extract_students' function with the 'original_data' dataframe and store the returned series in 'student_names'.
student_names = extract_students(original_data)

# Call the 'create_matrix' function with the 'data' dataframe and store the returned dataframe in 'matrix'.
matrix = create_matrix(data)


# Define a function named 'process_row' that takes a row and a predicted row as arguments and give row suitable to add in final matrix.
def process_row(row, predicted_row):
    # Make a copy of 'row' and store it in 'row_copy'.
    row_copy = row.copy()
    # Set the elements in 'predicted_row' that correspond to 1's in 'row' to 0.
    predicted_row[row == 1] = 0
    # Set the element in 'predicted_row' that corresponds to the name of 'row' to 0.
    predicted_row[row.index.get_loc(row.name)] = 0
    # While the sum of 'row_copy' is less than 30 and the sum of 'predicted_row' is not 0, do the following:
    while row_copy.sum() < 30 and predicted_row.sum() != 0:
        # Find the index of the maximum value in 'predicted_row' and store it in 'max_index'.
        max_index = predicted_row.argmax()
        # If the maximum value in 'predicted_row' is less than or equal to 0, return 'row_copy'.
        if predicted_row[max_index] <= 0:
            return row_copy
        # Set the element in 'row_copy' at 'max_index' to 1.
        row_copy[max_index] = 1
        # Set the element in 'predicted_row' at 'max_index' to 0.
        predicted_row[max_index] = 0
    # Return 'row_copy'.
    return row_copy

# Define a function named 'expected_matrix' that takes a matrix as an argument.
def expected_matrix(matrix):
    # Initialize an empty pandas dataframe with the same columns as 'matrix' and store it in 'expected_matrix'.
    expected_matrix = pd.DataFrame(index=[], columns=matrix.columns)

    # For each index in 'matrix', do the following:
    for i in matrix.index:
        # If the sum of the row at index 'i' in 'matrix' is 30, set the corresponding row in 'expected_matrix' to the row in 'matrix'.
        if np.sum(matrix.loc[i]) == 30:
            expected_matrix.loc[i] = matrix.loc[i]
        # Otherwise, drop the row at index 'i' from 'matrix' and store the result in 'df_except_current'. Then, set the corresponding row in 'expected_matrix' to the result of calling 'expected_row' with the row at index 'i' in 'matrix' and 'df_except_current'.
        else:
            df_except_current = matrix.drop(i)
            expected_matrix.loc[i] = expected_row(matrix.loc[i], df_except_current)
    # Return 'expected_matrix'.
    return expected_matrix


def get_coefficients(matrix, row):
    # Check if the matrix is empty
    if matrix.shape[0] == 0:
        # If it is, return 0 for all coefficients
        return np.zeros(row.shape[0])

    # Create a LinearRegression object
    lr = LinearRegression(fit_intercept=False)

    # Fit the model to your data
    lr.fit(matrix.T, row)

    # Return the coefficients
    return lr.coef_

# Define a function named 'expected_row' that takes a row and a matrix as arguments.
def expected_row(row, matrix):
    # Create a mask that is True where 'row' is 1 and False otherwise.
    mask = row == 1
    # Apply the mask to the columns of 'matrix' and store the result in 'useful_matrix'.
    useful_matrix = matrix.copy().loc[:, mask]

    # If 'row[mask]' is empty, return a pandas Series of zeros with the same index as 'row'.
    if row[mask].empty:
        return pd.Series(np.zeros(len(row)), index=row.index)

    # Call the 'get_coefficients' function with 'useful_matrix' and 'row[mask]' and store the returned coefficients in 'coefficients'.
    coefficients = get_coefficients(useful_matrix, row[mask])  
    # Return the result of calling 'process_row' with 'row' and the dot product of the transpose of 'matrix' and 'coefficients'.
    return process_row(row, np.dot(matrix.T, coefficients).T)

# Call the 'expected_matrix' function with the 'matrix' dataframe and store the returned dataframe in 'expected_matrix'.
expected_matrix = expected_matrix(matrix)

#make series student_names
student_names = pd.Series(student_names)

#now we will do reverse of load and preprocess data
#make a dataframe named impressions with index from original_data
impressions = pd.DataFrame(columns=original_data.columns)

#now we will fill the dataframe (30 values) 
#iterate for each value in original_data
for i in data.index:
    #fill the row with the names of students who impressed i
    impressions.loc[i] = expected_matrix.loc[i][expected_matrix.loc[i] == 1].index.tolist()

#apply student names and save data in csv
apply_student_names(impressions,student_names).to_csv("impressions.csv")

