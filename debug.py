import pandas as pd

# Load your CSV file
df = pd.read_csv('data/student-mat.csv')

# Get the shape of the DataFrame
rows, columns = df.shape

# Print the number of rows and columns
print("Number of rows:", rows)
print("Number of columns:", columns)
