# Predict Customer Churn with Clean Code

## Meets Specifications 

### Greetings Learner,

### Well done! , you have provided good quality codes by following best practices. Scripts run and successfully generated the outputs that are part of the requirements.

***Best part of your submission\***

```
 - Covered all required functionality and the codes run without error
 - Good work, code conforms to PEP 8 standard, the pylint run on both the scripts shows score higher than 7.
 - the logging functionality is used on all test functions and it correctly documents whether the test pass or if there is an error
 - log file is populated with the log messages based on the test outcomes
 - Model files are generated and stored in the correct folder, well done
```

### Regards

------

**Additional References**

- [MLOps: 10 Best Practices You Should Know](https://neptune.ai/blog/mlops-10-best-practices)
- [Testing Data Science and MLOps Code](https://microsoft.github.io/code-with-engineering-playbook/machine-learning/ml-testing/)



## Code Quality 

All the code written for this project should follow the [PEP 8 guidelines](https://www.python.org/dev/peps/pep-0008/). Objects have meaningful names and syntax. Code is properly commented and organized. Imports are correctly ordered.

Running the below can assist with formatting.

```
 autopep8 --in-place --aggressive --aggressive script.py
```

Then students should aim for a score exceeding 7 when using `pylint`

```
pylint script.py
```

### Good work, code conforms to PEP 8 standard, the pylint run on both the scripts shows score higher than 7.

[![Capture.PNG](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/3935679/1639914016/Capture.PNG)](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/3935679/1639914016/Capture.PNG)

------

**Additional References**

- [Python Code Quality: Tools & Best Practices](https://realpython.com/python-code-quality/)

The file contains a summary of the purpose and description of the project. Someone should be able to run the code by reading the README.

### Good work on providing details in the readme file, however, you can make it more informative

**Add details on Dependencies**

- For Instance, Sklearn need to be a higher version than in the default workspace to use plot_roc_curve
  and other dependencies which are required. Without this details, code cannot be run as-is
- You can add a requirement.txt file as well
- Also, you can add details on where the artifacts are stored, which folder should a reader need to navigate and check for the output.

All functions have a document string that correctly identifies the inputs, outputs, and purpose of the function. All files have a document string that identifies the purpose of the file, the author, and the date the file was created.

### Well done, you have updated the Doc String to covers the details on Input, output and purpose for all functions in both the scripts

------

**Additional References**

- [The Best of the Best Practices (BOBP) Guide for Python](https://gist.github.com/sloria/7001839)

## Testing & Logging 

Each function in `churn_script_logging_and_tests.py` is complete with tests for the input function. 

### Good work, you have constructed test functions in churn_script_logging_and_tests.py.

- you have correctly defined functionalities to test the inputs, provided assert statements to check the conditions.
- Appreciate the effort made to test all important functions such as encoding, feature engineering, train model etc.

------

**Additional References**

- [The writing and reporting of assertions in tests](https://docs.pytest.org/en/6.2.x/assert.html)

Each function in `churn_script_logging_and_tests.py` is complete with logging for if the function successfully passes the tests or errors.

### Well done, the logging functionality is used on all test functions and it correctly documents whether the test pass or if there is an error

All log information should be stored in a `.log` file, so it can be viewed post the run of the script.

### Good work , log file is populated with the log messages based on the test outcomes

The log messages should easily be understood and traceable that appear in the `.log` file. 

### Good work, log message for success and failures are easy to understand.

The README should inform a user how they would test and log the result of each function. 

Something similar to the below should produce the `.log` file with the result from running all tests.

```
ipython churn_script_logging_and_tests_solution.py
```

### Instructions for running test script provided in the readme file

## Save Images & Models 

Store result plots including at least one:

1. Univariate, quantitative plot
2. Univariate, categorical plot
3. Bivariate plot

### Well done, you have generated all required plots as part of EDA. The plots are stored in current folder once executed

- Required plots used including bivariate and multivariate

------

[![Capture7.PNG](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/3935679/1639914187/Capture7.PNG)](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/3935679/1639914187/Capture7.PNG)

Store result plots including:

1. ROC curves
2. Feature Importances

### Required plots are generated as part of the result.

- ROC curves
- Feature Importances

------

[![Capture1.PNG](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/3935679/1639914213/Capture1.PNG)](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/3935679/1639914213/Capture1.PNG)

Store at least two models. Recommended using `joblib` and storing models with `.pkl` extension.

### Model files are generated and stored in the correct folder, well done

------

**Additional References**

- [ONNX Model Zoo](https://github.com/onnx/models)
  Open Neural Network Exchange (ONNX) is an open standard format for representing machine learning models

## Problem Solving 

Code in `churn_library.py` completes the process for solving the data science process including:

1. EDA
2. Feature Engineering (including encoding of categorical variables)
3. Model Training 
4. Prediction
5. Model Evaluation

### Well done, the code - churn_library.py executes without any error and covers the following process

- EDA
- Feature Engineering (including encoding of categorical variables)
- Model Training
- Prediction
- Model Evaluation

Use one-hot encoding or mean of the response to fill in categorical columns. Currently, the notebook does this in an inefficient way that can be refactored by looping. Make this code more efficient using the same method as in the notebook or using one-hot encoding. Tip: Creating a list of categorical column names can help with looping through these items and create an easier way to extend this logic.

### Good work encoding of the categorical columns provided aligns with the requirement

------

**Additional References**

- [Smarter Ways to Encode Categorical Data for Machine Learning](https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159)