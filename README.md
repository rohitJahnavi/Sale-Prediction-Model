# Sale-Prediction-Model
Q. Sales Prediction Model:
* Description: Develop a regression model to predict future sales based
on historical data. This helps in forecasting and planning for inventory,
marketing, and budgeting.
* Why: Accurate sales predictions enable better decision-making and
resource allocation.

* Tasks:
  
▪ Gather historical sales data.

▪ Preprocess data (handling missing values, encoding categorical
variables).

▪ Given datasets 

▪ Train regression models (e.g., linear regression, random forest).

▪ Evaluate model performance and make predictions.

--------------------------------------------------------------------------------------------

Explanation:
1. Import Libraries:
  
   * pandas and numpy are used for data manipulation and numerical operations.

   * train_test_split from sklearn.model_selection splits the dataset into training and testing sets.

   * LinearRegression and RandomForestRegressor are the machine learning models used for prediction.

   * mean_absolute_error, mean_squared_error, and r2_score are evaluation metrics to assess the models' accuracy.

   * matplotlib and seaborn are for visualizations.

2. Load and Explore Data:

   * This loads the dataset from a CSV file named 'ecommerce_product_dataset.csv'.

   * df.info() provides basic information about the dataset (such as column types, non-null values).
  
   * df.head() shows the first few rows of the dataset to give an overview of its structure.

3. Data Preprocessing:
  
  # Handling Missing Values:

   * This fills any missing values with the forward-fill method (ffill), meaning it replaces missing values with the last 
     valid value.
  
  # Encoding Categorical Variables:

   * This checks if there are any categorical variables (data types of object), and if so, it uses one-hot encoding 
    (pd.get_dummies()) to convert them into numerical values.
  
   * drop_first=True drops the first category to avoid multicollinearity.

4. Feature and Target Selection:

   * The features (x) are all columns except Sales, which is the target variable (y).

5. Train-Test Split:

   * The dataset is split into training (80%) and testing (20%) sets.
   * random_state=42 ensures reproducibility of the split.
  
6. Model Training:
   
   * A dictionary is created with two models: Linear Regression and Random Forest Regressor.
  
   * For each model, the fit() method is called to train the model using the training data (x_train, y_train).
  
 7. Model Evaluation:

    *  The models are evaluated using three metrics:
      
    * Mean Absolute Error (MAE): Measures the average magnitude of errors in predictions.
      
    * Mean Squared Error (MSE): Measures the average squared difference between actual and predicted values.
      
    * R² Score: Indicates how well the model explains the variance in the data (higher is better).
   
    * A results DataFrame is created from the dictionary to display the evaluation metrics for both models.
   
  8. Visualizing Results:

     * The model with the highest R² score is selected as the best-performing model.
    
     * Predictions are made using the best model (best_model).
    
     * A scatter plot is generated to show the actual vs predicted sales for the best-performing model.
    
     * The red dashed line represents perfect predictions (where actual equals predicted).
    
  9. Save the Model for Future Use:

      * The best model is saved to a file named 'best_sales_model.pkl' using joblib, which can be loaded later for future 
        predictions.



