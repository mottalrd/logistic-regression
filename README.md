# Installation

This code has been tested with Python 3.5.0

First create a virtual environment 

    $: pyenv versions
      system
      2.7.11rc1
      3.5.0
    $: pyenv virtualenv 3.5.0 logistic-regression
    $: pip install -r requirements.txt

You can now load the LinearRegression classes in a console:

    >>> import imp  
    >>> logistic_regression = imp.load_source('logistic_regression', '../lib/logistic_regression.py')
    >>> from logistic_regression import *
    >>> from sklearn import datasets 
    >>> x, y = datasets.make_classification(500, n_features=1, n_informative=1, n_redundant=0, n_clusters_per_class=1)
    >>> binomial = Binomial()
    >>> binomial.fit(x, y)

Check the experiments folder to see the code in action with a Jupyter notebook.

# Running the tests

    py.test

