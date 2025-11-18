### Loading the Required Packages

To proceed with reading the data, performing numerical operations, and visualizing relationships, we need the following libraries:

- **pandas** – for reading and handling CSV files  
- **numpy** – for working with arrays and numerical transformations  
- **matplotlib** – for plotting and visual exploration of the data  

**Installation (run once):**

```bash
pip install pandas numpy matplotlib
```
### Dataset Introduction – The Classic Advertising Dataset  
This is the famous Advertising dataset from the book Introduction to Statistical Learning (ISLR).  
- All monetary values are in thousands of dollars  
- TV – advertising budget spent on television  
- Radio – advertising budget spent on radio  
- Newspaper – advertising budget spent on newspapers  
- Sales – actual sales (target variable)  



```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
```

### Level 2: Loading and Initial Inspection of the Dataset  
**Loading the Dataset**  
Reads the CSV file and stores it in a pandas DataFrame called `df`.  
(If your file has a different name or path, adjust the string accordingly.)


```python
df = pd.read_csv("/home/pyrz-tech/Desktop/MachineLearning/advertising.csv")
```

**Preview the First Rows**  
‍`df.head()`Displays the first 5 rows of the DataFrame, allowing a quick visual verification of the loaded data.


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TV</th>
      <th>Radio</th>
      <th>Newspaper</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>230.1</td>
      <td>37.8</td>
      <td>69.2</td>
      <td>22.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44.5</td>
      <td>39.3</td>
      <td>45.1</td>
      <td>10.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17.2</td>
      <td>45.9</td>
      <td>69.3</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>151.5</td>
      <td>41.3</td>
      <td>58.5</td>
      <td>16.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>180.8</td>
      <td>10.8</td>
      <td>58.4</td>
      <td>17.9</td>
    </tr>
  </tbody>
</table>
</div>



**Dataset Dimensions**   
`df.shape` Returns the total number of rows and columns in the dataset.


```python
df.shape
```




    (200, 4)



**Column Information**  
`df.info()` Shows column names, data types, non-null counts, and memory usage.


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 200 entries, 0 to 199
    Data columns (total 4 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   TV         200 non-null    float64
     1   Radio      200 non-null    float64
     2   Newspaper  200 non-null    float64
     3   Sales      200 non-null    float64
    dtypes: float64(4)
    memory usage: 6.4 KB


**Descriptive Statistics**  
`df.describe()` Provides summary statistics (count, mean, std, min, quartiles, max) for numerical columns.



```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TV</th>
      <th>Radio</th>
      <th>Newspaper</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>200.000000</td>
      <td>200.000000</td>
      <td>200.000000</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>147.042500</td>
      <td>23.264000</td>
      <td>30.554000</td>
      <td>15.130500</td>
    </tr>
    <tr>
      <th>std</th>
      <td>85.854236</td>
      <td>14.846809</td>
      <td>21.778621</td>
      <td>5.283892</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.700000</td>
      <td>0.000000</td>
      <td>0.300000</td>
      <td>1.600000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>74.375000</td>
      <td>9.975000</td>
      <td>12.750000</td>
      <td>11.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>149.750000</td>
      <td>22.900000</td>
      <td>25.750000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>218.825000</td>
      <td>36.525000</td>
      <td>45.100000</td>
      <td>19.050000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>296.400000</td>
      <td>49.600000</td>
      <td>114.000000</td>
      <td>27.000000</td>
    </tr>
  </tbody>
</table>
</div>




### Quick Summary of What We’ve Seen So Far  
After running the basic checks, we confirmed:  
- Shape: 200 rows × 4 columns
- All feature columns (TV, Radio, Newspaper) and the target (Sales) are of type float64  
- No missing values  

### Visual Inspection of Individual Feature–Sales Relationships  
We now carefully examine the relationship between each advertising channel and Sales using individual scatter plots with regression lines. The goal is to visually assess:  
- Strength of the linear relationship  
- Density and spread of points around the fitted line  
- Which feature appears to have the strongest and most compact linear relationship with Sales

#### Visual Inspection Using Matplotlib’s scatter() Method  
We now plot the relationship between each advertising feature and Sales using pure **matplotlib.scatter()** (no seaborn regplot) so that we can fully control the appearance and clearly see the raw data points.


```python
plt.scatter(df.TV, df.Sales)
```




    <matplotlib.collections.PathCollection at 0x711d130f0ef0>




    
![png](media/output_14_1.png)
    



```python
plt.scatter(df.Radio, df.Sales)
```




    <matplotlib.collections.PathCollection at 0x711d1335e3f0>




    
![png](media/output_15_1.png)
    



```python
plt.scatter(df.Newspaper, df.Sales)
```




    <matplotlib.collections.PathCollection at 0x711d132ddeb0>




    
![png](media/output_16_1.png)
    


### Visual Analysis Summary and Feature Selection for Simple Linear Regression  

As observed in the scatter plots above:  

- All three advertising channels (TV, Radio, Newspaper) show a positive relationship with Sales.  
- The TV advertising budget exhibits the strongest, most densely clustered, and clearest linear relationship with Sales.  
- The TV feature has the steepest slope, the tightest spread around the trend, and the fewest apparent outliers.  

Therefore, based on visual inspection and exploratory analysis, we select TV as the single predictor variable for our Simple Linear Regression model.

#### Selected Feature  
**Feature:**   `TV`   
**Target:**   `Sales`

### Creating a Clean Subset for Focused Analysis  

To work more cleanly and concentrate only on the selected feature (TV) and the target (Sales), we create a new DataFrame called `cdf` (clean DataFrame) containing just these two columns.

From now on, we will perform all subsequent steps (visualization, modeling, evaluation) using `cdf` instead of the full `df`. This keeps our workspace focused and readable.


```python
cdf = df[['TV', 'Sales']]
```

### Train-Test Split (Manual Random Split)  

We now split the clean dataset (`cdf`) into training and test sets using a simple random mask.  
Approximately 80 % of the data will be used for training and the remaining 20 % for testing.

This is a common manual approach when we want full control over the splitting process without importing `train_test_split` from scikit-learn.

`train` and `test` DataFrames are ready for model training and evaluation.  


```python
msk = np.random.rand(len(cdf)) < 0.8
train = cdf[msk]
test = cdf[~msk]

print(f'msk => {msk[:4]} ...')
print(f'train => {train.head()}')
print('...')
print(f'test => {test.head()} ...')
print('...')
print(f'len(train) => {len(train)}')
print(f'len(test) => {len(test)}')
```

    msk => [ True  True  True False] ...
    train =>       TV  Sales
    0  230.1   22.1
    1   44.5   10.4
    2   17.2   12.0
    5    8.7    7.2
    6   57.5   11.8
    ...
    test =>        TV  Sales
    3   151.5   16.5
    4   180.8   17.9
    8     8.6    4.8
    9   199.8   15.6
    10   66.1   12.6 ...
    ...
    len(train) => 156
    len(test) => 44


 ### Visualizing the Training and Test Sets on the Same Plot  

Before training the model, we plot both the training and test data points on the same scatter plot (with different colors) to visually confirm that:  
- The split appears random  
- Both sets cover the same range of TV and Sales values  
- There is no systematic bias in the split



```python
plt.scatter(train.TV, train.Sales)
plt.scatter(test.TV, test.Sales, color='green')
```




    <matplotlib.collections.PathCollection at 0x711d12010980>




    
![png](media/output_23_1.png)
    


 #### Converting Training Data to NumPy Arrays  

For the scikit-learn LinearRegression model, we need the feature and target variables as NumPy arrays (or array-like objects).  
We use `np.asanyarray()` to convert the pandas columns from the training set into the required format.


```python
train_x = np.asanyarray(train[['TV']])
train_y = np.asanyarray(train[['Sales']])
```

### Fitting the Simple Linear Regression Model  

We now import the LinearRegression class from scikit-learn, create a model instance, and train it using the prepared training arrays (train_x and train_y).

After running, the simple linear regression model is fully trained using only the TV advertising budget to predict Sales.  
The coefficient tells us how much Sales increases (in thousand units) for every additional thousand dollars spent on TV advertising.


```python
from sklearn.linear_model import LinearRegression
```


```python
reg = LinearRegression()
reg.fit(train_x, train_y)
```

### Visualizing the Fitted Regression Line  

In this step we plot the training data points together with the regression line found by the model. This allows us to visually verify that the fitted line reasonably captures the linear relationship between TV advertising and Sales.  

The line is drawn using the learned parameters:  
- `model.coef_[0]` → slope of the line  
- `model.intercept_` → y-intercept  


```python
plt.scatter(train_x, train_y)
plt.plot(train_x, reg.coef_[0][0] * train_x + reg.intercept_[0], '-g')
```




    [<matplotlib.lines.Line2D at 0x711d1213bc50>]




    
![png](media/output_30_1.png)
    


### Preparing Test Data and Making Predictions  

We convert the test set to NumPy arrays (required format for scikit-learn) and use the trained model to predict Sales values for the test observations.


```python
test_x = np.asanyarray(test[['TV']])
test_y = np.asanyarray(test[['Sales']])
predict_y = np.asanyarray(reg.predict(test_x))
```

### Evaluating Model Performance with R² Score  

We import the `r2_score` metric from scikit-learn to measure how well our Simple Linear Regression model performs on the test set.  
The R² score (coefficient of determination) tells us the proportion of variance in Sales that is explained by the TV advertising budget.
    
- R² ≈ 1.0 → perfect fit  
- R² ≈ 0 → model explains nothing  


```python
from sklearn.metrics import r2_score
```

### Computing and Displaying the R² Score  

We use the imported `r2_score` function to calculate the coefficient of determination on the test data and print the result directly.

This single line gives us the final performance metric: the higher the value (closer to 1.0),
the better our simple linear regression model using only TV advertising explains the variation in Sales.


```python
print(f'r^2 score is : {r2_score(test_y, predict_y)}')
```

    r^2 score is : 0.8674734235783073

