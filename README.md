# cse512_project
Code and links to data related to my project for CSE 512 Machine Learning @ SUNY Stony Brook.

## Purpose
To build more concrete intuition on how standard classification algorithms work. Inspired by Dr. Charles Isabel on the Lex Fridman podcast when discussing
how he teaches machine learning to students:

_"Implement these algorithms - linear regression, neural networks, SVMs, whatever - and by 'implement' I mean 'steal the code'. You get zero points for getting the code to work because I am not interested in that. Run each algorithm on two interesting datasets. By interesting, I mean it will reveal the differences between [each of] the algorithms, which are presumably all the same. The datasets will be interesting together if each dataset shows different differences for a single algorithm. This will teach you how much the data matters, what it can tell you, and what questions it answers, especially for the questions that are different than the ones you originally asked"_

## Data
The data could not be loaded to the repo due to its size, but here is where each dataset was sourced from.

| Dataset Name | Label Type | Label Balance | Dataset size |
| ------------ |  --------  | ----- | ---- |
| [Penguins](https://inria.github.io/scikit-learn-mooc/python_scripts/trees_dataset.html) | Multiclass | Unbalanced | 344 observations |
| [Heart Disease](https://www.kaggle.com/fedesoriano/heart-failure-prediction) | Binary | Balanced | 918 observations |
| [MNIST Fashion](https://www.kaggle.com/zalando-research/fashionmnist) | Multiclass | Balanced | 60,000 observations |
| [Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud) | Binary | Unbalanced | 284,807 observations |


## Visuals
Each script writes out some `.png` files for the analysis. These get written into directories that you, dear reader, will need to create yourself.

## References
 - [Characterization of Classification Algorithms - J.Gama and P.Brazdil, LIACC, University of Porto](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.25.202&rep=rep1&type=pdf)
 - [chapter 2: Methods for comparison, R.J. Henery, University of Strathclyde.](http://www1.maths.leeds.ac.uk/~charles/statlog/)
 - [An overview of classification algorithms for imbalanced datasets - Vaishali Ganganwar, Army Institute of Technology, Pune](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.413.3344&rep=rep1&type=pd)
 - [Top US Causes of Death](https://www.cdc.gov/nchs/fastats/leading-causes-of-death.htm)
 - [Top US Causes of Medical Expenses](https://www.cdc.gov/chronicdisease/about/costs/index.htm)
 - [Heart disease percentages in USA](https://www.cdc.gov/heartdisease/facts.htm)

