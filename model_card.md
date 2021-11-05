# Model Card

For additional information on model cards see the [Model Card paper](https://arxiv.org/pdf/1810.03993.pdf)

## Model Details
The model used was scikit-learn's [gradient boosting classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html).

Hyper parameter tuning was carried out using [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
Including the defaults, the parameters used are:
  - learning_rate: 0.1
  - max_depth: 5
  - n_estimators: 100
## Intended Use
The model is _only_ intended for the demonstration of deploying and serving a machine learning model.

## Training Data
80% of the data was used for training.
## Evaluation Data
20% of the data was used for testing/evaluation.
## Metrics
The metrics used to evaluate the model are:
  - f1 score
  - precision
  - recall
## Ethical Considerations
The dataset is quite outdated since it's from 1994. It does not provide any insight into current income with respect to demographics. The data is also not representative enough of the demographics within the dataset. See this [bias report](http://aequitas.dssg.io/audit/3pc04g0h/adult_rf_binary/report-1.html) for more details.
