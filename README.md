# Data Cleaning & Preprocessing

## Objective

The objective of this task is to learn how to clean and prepare raw data for Machine Learning.

## Tools & Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Dataset

**Titanic Dataset (`titanic.csv`)**

The Titanic dataset contains information about passengers, such as:

* Age
* Sex
* Fare
* Embarked
* Survived
* Passenger Class

## Steps Performed

### 1. Import Dataset and Explore Data

The dataset is loaded using Pandas.

```python
import pandas as pd

df = pd.read_csv("titanic.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())
```

The following functions are used to understand the dataset:

* `head()` → Displays the first five rows.
* `info()` → Displays column names, data types, and non-null values.
* `isnull().sum()` → Counts missing values in each column.

---

### 2. Handle Missing Values

Missing values are handled using imputation.

For the `Age` column, missing values are replaced with the mean age:

```python
df.Age = df.Age.fillna(df.Age.mean())
```

For the `Embarked` column, missing values can be replaced using the mode:

```python
df.Embarked = df.Embarked.fillna(df.Embarked.mode()[0])
```

The `Cabin` column can be removed if it contains a large number of missing values:

```python
df = df.drop("Cabin", axis=1)
```

---

### 3. Convert Categorical Data into Numerical Data

Machine Learning models generally require numerical data.

The `Sex` column is converted into numerical values:

```python
df.Sex = df.Sex.map({"male": 1, "female": 0})
```

Here:

* Male → `1`
* Female → `0`

The `Embarked` column can be converted using one-hot encoding:

```python
df = pd.get_dummies(df, columns=["Embarked"], dtype=int)
```

---

### 4. Standardize Numerical Features

Numerical features such as `Age` and `Fare` are standardized using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])
```

Standardization converts the values to a common scale with approximately:

* Mean = `0`
* Standard deviation = `1`

---

### 5. Visualize Outliers

Boxplots are used to identify possible outliers.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df.Fare)
plt.title("Fare Outliers")
plt.show()
```

Values appearing outside the normal range in the boxplot may be potential outliers.

---

### 6. Remove Outliers Using IQR

The Interquartile Range (IQR) method is used to detect and remove outliers.

```python
Q1 = df.Fare.quantile(0.25)
Q3 = df.Fare.quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df.Fare >= lower) & (df.Fare <= upper)]
```

### IQR Formula

```text
IQR = Q3 - Q1

Lower Limit = Q1 - 1.5 × IQR

Upper Limit = Q3 + 1.5 × IQR
```

Values below the lower limit or above the upper limit are considered outliers.

---

## Final Verification

After preprocessing, the dataset is checked again:

```python
print("After preprocessing:")
print(df.head())
print(df.isnull().sum())
```

This verifies that the dataset has been cleaned and missing values have been handled.

## Conclusion

The Titanic dataset was successfully cleaned and prepared for Machine Learning by:

1. Exploring the dataset.
2. Handling missing values.
3. Encoding categorical features.
4. Standardizing numerical features.
5. Visualizing outliers using boxplots.
6. Removing outliers using the IQR method.

The resulting dataset is now better prepared for further Machine Learning tasks.
