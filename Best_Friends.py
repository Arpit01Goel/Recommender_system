#Import necessary libraries
#pandas to use DataFrame
import pandas as pd
#numpy for numerical operations
import numpy as np
#warnings to ignore warnings
import warnings

# Ignore all warning messages for cleaner output
warnings.filterwarnings("ignore")

# Function to load and preprocess the data
def load_and_preprocess_data(filename):
    # Load the data from a CSV file
    data = pd.read_csv(filename, index_col=1).drop("Timestamp", axis=1)

    # Make a copy of the original data for later use
    original_data = data.copy()

    # Preprocess the data by transforming all entries to uppercase and trimming unnecessary parts
    #remove name from values and keep entry no only
    data = data.applymap(lambda x: str(x)[-11:].upper())
    #remove your from column name
    data.columns = data.columns.map(lambda x: x[5:])
    #make index in capital letters
    data.index = data.index.map(lambda x : x[0:11].upper())
    #give indexes a name
    data.index.name = "Entry Number"

    # Return the preprocessed data and the original data
    return data, original_data

# Load and preprocess the data from the "Impression_Network.csv" file
data, original_data = load_and_preprocess_data("Impression_Network.csv")

# Function to create a matrix from the data
def create_matrix(data):
    # Get a list of unique students from the data
    students = list(set(data.values.flatten()))

    # Remove the "NAN" entry from the list of students
    students.remove("NAN")

    # Create a dataframe with the students as both the index and columns, and fill it with zeros
    matrix = pd.DataFrame(index=students, columns=students)
    #make all values 0
    matrix[:] = 0

    # For each student, mark the students they have interacted with in the matrix
    for i in data.index:
        #make a series of entry no of students who impressed i
        unique_columns = pd.Series(data.loc[i].values).drop_duplicates().values
        #make those values 1 ( adjcency matrix)
        matrix.loc[i, unique_columns] = 1

    # Remove the "NAN" column from the matrix
    matrix.drop("NAN", axis=1, inplace=True)
    
    # Return the created matrix
    return matrix

# Create a matrix from the data
matrix=create_matrix(data)

# Function to calculate the impressing vector for a given node in the matrix
def impressing_vector(matrix, node):
    # Transpose the matrix because its impressing vector, not impressed vector
    matrix=matrix.T

    # If the node has no connections, return a series of zeros. as impressing ability is 0
    if matrix.loc[node].sum() == 0:
        #return the series of 0
        return pd.Series(0, index=matrix.index)

    # Create a reduced matrix by removing the node's row and column
    reduced_matrix=matrix.drop(node,axis=0).drop(node,axis=1)

    # Initialize the final answer and the current distribution
    final_answer=pd.Series(0,index=matrix.index)
    #we will calculate vector of 142 dimensions( 143 students) then add self dimension
    final_row=matrix.loc[node].drop(node)
    #make it 0 in all directions
    final_row=0*final_row

    # Create a mask for the nodes that the current node is impressing
    mask=matrix.loc[node].drop(node)==1

    # Distribute the initial influence of the node to its connections
    current_distribution = final_row.copy()
    #distribute the points
    current_distribution[mask] = 500 / mask.sum()

    # Initialize the new distribution
    new_distribution = final_row.copy()

    # Perform the influence propagation for 5 iterations( whole graph will be covered in 5 iterations )
    for _ in range(3):
        # for each step, distribute the points to the students who are impressed by the current student
        for i in range(len(current_distribution)):
            #if node has points and have students impressed by him/her
            if current_distribution[i] != 0 and len(reduced_matrix.iloc[i]) != 0:
                #get the index of the students
                index_label = current_distribution.index[i]
                #get the mask of the students
                mask = reduced_matrix.loc[index_label].values.astype(bool)
                #get the indices of the students
                indices = new_distribution.index[mask]
                #distribute the points
                new_distribution.loc[indices] += current_distribution[i] / len(reduced_matrix.iloc[i])

        # Update the final row and the current distribution
        final_row += current_distribution
        current_distribution = new_distribution.copy()

    # Update the final row and the final answer
    final_row += current_distribution
    #put the collected values in final_answer ( dim 143)
    final_answer.update(final_row)
    
    # Replace any NaN values in the final answer with 0
    final_answer.fillna(0,inplace=True)

    # Return the final answer
    return final_answer

def impressed_vector(matrix,node):
        # Transpose the matrix because its impressing vector, not impressed vector
    matrix=matrix

    # If the node has no connections, return a series of zeros. as impressing ability is 0
    if matrix.loc[node].sum() == 0:
        #return the series of 0
        return pd.Series(0, index=matrix.index)

    # Create a reduced matrix by removing the node's row and column
    reduced_matrix=matrix.drop(node,axis=0).drop(node,axis=1)

    # Initialize the final answer and the current distribution
    final_answer=pd.Series(0,index=matrix.index)
    #we will calculate vector of 142 dimensions( 143 students) then add self dimension
    final_row=matrix.loc[node].drop(node)
    #make it 0 in all directions
    final_row=0*final_row

    # Create a mask for the nodes that the current node is impressing
    mask=matrix.loc[node].drop(node)==1

    # Distribute the initial influence of the node to its connections
    current_distribution = final_row.copy()
    #distribute the points
    current_distribution[mask] = 500 / mask.sum()

    # Initialize the new distribution
    new_distribution = final_row.copy()

    # Perform the influence propagation for 5 iterations( whole graph will be covered in 5 iterations )
    for _ in range(3):
        # for each step, distribute the points to the students who are impressed by the current student
        for i in range(len(current_distribution)):
            #if node has points and have students impressed by him/her
            if current_distribution[i] != 0 and len(reduced_matrix.iloc[i]) != 0:
                #get the index of the students
                index_label = current_distribution.index[i]
                #get the mask of the students
                mask = reduced_matrix.loc[index_label].values.astype(bool)
                #get the indices of the students
                indices = new_distribution.index[mask]
                #distribute the points
                new_distribution.loc[indices] += current_distribution[i] / len(reduced_matrix.iloc[i])

        # Update the final row and the current distribution
        final_row += current_distribution
        current_distribution = new_distribution.copy()

    # Update the final row and the final answer
    final_row += current_distribution
    #put the collected values in final_answer ( dim 143)
    final_answer.update(final_row)
    
    # Replace any NaN values in the final answer with 0
    final_answer.fillna(0,inplace=True)

    # Return the final answer
    return final_answer


# Function to find the best friends in the matrix
def best_friends(matrix):
    # Initialize a dictionary to store the impressing and impressed unit vectors for each student
    best_friends_dict={}

    # Calculate the impressing and impressed unit vectors for each student and store them in the dictionary
    for i in matrix.index:
        #store the pair of matirx in tuple
        best_friends_dict[i]=(impressing_vector(matrix,i),impressed_vector(matrix,i))

    # Return the dictionary
    return best_friends_dict

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

# Calculate the impressing and impressed unit vectors for each student
vectors_dict=best_friends(matrix)

# Initialize a dataframe to store the distances between each pair of students
dist_matrix=pd.DataFrame(index=matrix.index,columns=matrix.index)

# Calculate the distances between each pair of students
#iterate first iterator on all students
for i in matrix.index:
    #iterate 2nd iterator on all students
    for j in matrix.index:
        #calculate the distance between the students
        dist_matrix.loc[i,j]=np.sqrt(np.square((vectors_dict[i][0]-vectors_dict[j][0])).sum()+np.square(vectors_dict[i][1]-vectors_dict[j][1]).sum())

# Initialize a dataframe to store the best friends of each student
answer=pd.DataFrame(index=matrix.index,columns=[i for i in range(0,len(matrix.index))])

# For each student, sort their distances to other students and store the sorted indices in the answer dataframe
for i in range(len(matrix.index)):
    #use in build sort function to sort the distances
    answer.iloc[i]=dist_matrix.iloc[i].sort_values().index


# Extract the student names from the original data
student_names=extract_students(original_data)

# Apply the student names to the answer dataframe
answer=apply_student_names(answer,student_names)

#take top 8 students
answer=answer.iloc[:,1:9]    #since 0 is same as index 

# Save the answer dataframe to a CSV file
answer.to_csv("Best_Friends.csv")

# Print a message to indicate that the best friends of each student have been stored in the "Best_Friends.csv" file
print("Best Friends of each student is stored in Best_Friends.csv file")



