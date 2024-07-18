# create the list data
data = [50,50,47,97,49,3,53,42,26,74,82,62,37,15,70,27,36,35,48,52,63,64]
print(data)

# import library
import numpy as np

# making array by numpy library
grades = np.array(data)
print(grades)

# numpy arrays vs list ?
print (type(data),'x 2:', data * 2) # this is *2 on list to 50 -> 50,50
print('---')
print (type(grades),'x 2:', grades * 2) # this is numpy array take multiple numerical  50 -> 100
grades.shape
grades[0]
grades.mean()

# Define an array of study hours
study_hours = [10.0,11.5,9.0,16.0,9.25,1.0,11.5,9.0,8.5,14.5,15.5,
               13.75,9.0,8.0,15.5,8.0,9.0,6.0,10.0,12.0,12.5,12.0]

# Create a 2D array (an array of arrays)
student_data = np.array([study_hours, grades])

# display the array
student_data

# Show shape of 2D array
student_data.shape

# Show the first element of the first element
student_data[0][0]

# Get the mean value of each sub-array
avg_study = student_data[0].mean()
avg_grade = student_data[1].mean()

print('Average study hours: {:.2f}\nAverage grade: {:.2f}'.format(avg_study, avg_grade))

# Pandas
import pandas as pd

# this below is key:value matach {'Name':[list of data]}
df_students = pd.DataFrame({'Name': ['Dan', 'Joann', 'Pedro', 'Rosie', 'Ethan', 'Vicky', 'Frederic', 'Jimmie', 
                                     'Rhonda', 'Giovanni', 'Francesca', 'Rajab', 'Naiyana', 'Kian', 'Jenny',
                                     'Jakeem','Helena','Ismat','Anila','Skye','Daniel','Aisha'],
                            'StudyHours':student_data[0],
                            'Grade':student_data[1]})

df_students 

# Get the data for index value 5
df_students.loc[5]
# Get the rows with index values from 0 to 5
df_students.loc[0:5]
# Get data in the first five rows
df_students.iloc[0:5]
df_students.iloc[0,[1,2]]
df_students.loc[0,'Grade']

df_students.loc[df_students['Name']=='Aisha']
# or Actually, you don't need to explicitly use the loc method to do this. 
# You can simply apply a DataFrame filtering expression, like this:
df_students[df_students['Name']=='Aisha']

df_students.query('Name=="Aisha"')
df_students[df_students.Name == 'Aisha']

!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/grades.csv
df_students = pd.read_csv('grades.csv',delimiter=',',header='infer')
df_students.head()

# Handling missing values
df_students.isnull()
df_students.isnull().sum()

df_students[df_students.isnull().any(axis=1)]

# mean Imputation
df_students.StudyHours = df_students.StudyHours.fillna(df_students.StudyHours.mean())
df_students

df_students = df_students.dropna(axis=0, how='any')
df_students

# Get the mean study hours using to column name as an index
mean_study = df_students['StudyHours'].mean()

# Get the mean grade using the column name as a property (just to make the point!)
mean_grade = df_students.Grade.mean()

# Print the mean study hours and mean grade
print('Average weekly study hours: {:.2f}\nAverage grade: {:.2f}'.format(mean_study, mean_grade))

# Get students who studied for the mean or more hours
df_students[df_students.StudyHours > mean_study]
# What was their mean grade?
df_students[df_students.StudyHours > mean_study].Grade.mean()

passes  = pd.Series(df_students['Grade'] >= 60)
df_students = pd.concat([df_students, passes.rename("Pass")], axis=1)

df_students

print(df_students.groupby(df_students.Pass).Name.count())

print(df_students.groupby(df_students.Pass)[['StudyHours', 'Grade']].mean())

# Create a DataFrame with the data sorted by Grade (descending)
df_students = df_students.sort_values('Grade', ascending=False)

# Show the DataFrame
df_students

# Visualize data with Matplotlib
import pandas as pd

# Load data from a text file
!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/grades.csv
df_students = pd.read_csv('grades.csv',delimiter=',',header='infer')

# Remove any rows with missing data
df_students = df_students.dropna(axis=0, how='any')

# Calculate who passed, assuming '60' is the grade needed to pass
passes  = pd.Series(df_students['Grade'] >= 60)

# Save who passed to the Pandas dataframe
df_students = pd.concat([df_students, passes.rename("Pass")], axis=1)


# Print the result out into this notebook
df_students 

# Ensure plots are displayed inline in the notebook
%matplotlib inline

from matplotlib import pyplot as plt

# Create a bar plot of name vs grade
plt.bar(x=df_students.Name, height=df_students.Grade)

# Display the plot
plt.show()

# Create a bar plot of name vs grade
plt.bar(x=df_students.Name, height=df_students.Grade, color='orange')

# Customize the chart
plt.title('Student Grades')
plt.xlabel('Student')
plt.ylabel('Grade')
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
plt.xticks(rotation=90)

# Display the plot
plt.show()

# Create a Figure
fig = plt.figure(figsize=(8,3))

# Create a bar plot of name vs grade
plt.bar(x=df_students.Name, height=df_students.Grade, color='orange')

# Customize the chart
plt.title('Student Grades')
plt.xlabel('Student')
plt.ylabel('Grade')
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
plt.xticks(rotation=90)

# Show the figure
plt.show()

# Create a figure for 2 subplots (1 row, 2 columns)
fig, ax = plt.subplots(1, 2, figsize = (10,4))

# Create a bar plot of name vs grade on the first axis
ax[0].bar(x=df_students.Name, height=df_students.Grade, color='orange')
ax[0].set_title('Grades')
ax[0].set_xticklabels(df_students.Name, rotation=90)

# Create a pie chart of pass counts on the second axis
pass_counts = df_students['Pass'].value_counts()
ax[1].pie(pass_counts, labels=pass_counts)
ax[1].set_title('Passing Grades')
ax[1].legend(pass_counts.keys().tolist())

# Add a title to the Figure
fig.suptitle('Student Data')

# Show the figure
fig.show()

import matplotlib.pyplot as plt

# Create a figure
fig = plt.figure()

# Create a 2x2 grid of subplots
ax1 = plt.subplot(2, 2, 1)  # First subplot
ax2 = plt.subplot(2, 2, 2)  # Second subplot
ax3 = plt.subplot(2, 2, 3)  # Third subplot
ax4 = plt.subplot(2, 2, 4)  # Fourth subplot

# Plot something on each subplot
ax1.plot([1, 2, 3], [1, 4, 9])
ax2.plot([1, 2, 3], [1, 2, 3])
ax3.plot([1, 2, 3], [1, 3, 6])
ax4.plot([1, 2, 3], [1, 5, 10])

# Show the plot
plt.show()
