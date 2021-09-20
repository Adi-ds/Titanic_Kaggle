This dataset is based on the passengers who boarded Titanic from the three ports. This dataset has been obtained from the [legendary Titanic ML competition of Kaggle](https://www.kaggle.com/c/titanic/).
## Data Dictionary
According to [Kaggle](https://www.kaggle.com/c/titanic/data?select=train.csv) :

Variable    | Definition          | Key
------------|---------------------|-------------
PassengerId | ID of the Passenger |
Survived    | Survival            |0 = No; 1 = Yes
Pclass      | Ticket Class        |1 = 1st; 2 = 2nd; 3 = 3rd
Name        | Name of the Passenger |
Sex         | Sex                 |
Age         | Age in years        |
Sibsp       | Number of Siblings/Spouses Aboard |
Parch       | Number of Parents/Children Aboard |
Ticket      | Ticket Number       |
Fare        | Passenger Fare      |
Cabin       | Cabin Number        |
Embarked    | Port of Embarkation |C = Cherbourg; Q = Queenstown; S = Southampton

## Parts of this project
- Part 1 : Data Cleaning and Feature Engineering
- Part 2 : EDA and Visualization

## Idea of Progress
The RMS Titanic started the transatlantic journey from Southampton, UK to New York, USA on 10th April 1992. But on the night of 14 April 1992 the ship hit an iceberg and finally at 2:20 AM, 15 April 1912, the ship foundered with well over one thousand people still aboard. This dataset contains the details of the passengers who boarded Titanic. This work mainly aims on Exploratory Data Analysis along with Feature Engineering and Data Cleaning. With Feature Engineering the names of the passengers are divided into Title, First Name and Last Name. The Ages of the passengers are also grouped using bins. The Fares of each person has also been calculated. In Data Cleaning part, Title and Last Name of the passengers have been used for Missing value treatment on Age, Embarked and Fare columns. Now, Exploratory Data Analysis (EDA) refers to the step-by-step process of constantly investigating the data on some assumptions by the help of graphical representations, tabular representations and statistical hypothesis testing to uncover anomaly and discover patterns and trends. Here, EDA has been performed to find the conditions on which the passengers survived. Four conditions were seen to affect the survival of a passenger – Sex, Age, Social Status and Family. Along with these, 63 families were also found to board the Titanic. 

### Reading Order
1. [Titanic History](https://github.com/Adi-ds/Titanic_Kaggle/blob/main/Titanic%20History.md)
2. [Data_Cleaning_and_Feature_Engineering](https://github.com/Adi-ds/Titanic_Kaggle/blob/main/Data_Cleaning_and_Feature_Engineering.ipynb)
3. [Titanic_EDA_Visualizations](https://github.com/Adi-ds/Titanic_Kaggle/blob/main/Titanic_EDA_Visualizations.ipynb)
4. [Titanic_Decision_Tree](https://github.com/Adi-ds/Titanic_Kaggle/tree/main/Titanic_Decision_Tree)
5. [Support Vector Machine](https://github.com/Adi-ds/Titanic_Kaggle/tree/main/Support%20Vector%20Machine)
6. [Conclusion](https://github.com/Adi-ds/Titanic_Kaggle/blob/main/Conclusion.md)
