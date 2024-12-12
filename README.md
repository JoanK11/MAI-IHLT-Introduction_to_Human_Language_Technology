# Semantic Similarity Prediction Project

## Overview

This project aims to compute the semantic similarity between pairs of sentences using supervised learning approaches. It involves preprocessing datasets, extracting lexical and syntactic features, training machine learning models, and evaluating their performance.

The methodology is based on SemEval-2012 Task 6, a benchmark for semantic similarity tasks. Three types of models (Linear Regression, Random Forest, and Gradient Boosting) are implemented to predict similarity scores. Feature importance and model-specific feature analyses are conducted to identify the contributions of lexical and syntactic features.

---

## Project Structure

The project is structured into two main notebooks: first, `feature_extraction.ipynb`, and then `training_and_results.ipynb`. In the `datasets` folder, we have all the datasets in `.txt` and `.csv` formats. In the `features` folder, we have the features already extracted with the first notebook. In the `models` folder, we have the best models found in the second notebook. In the `papers` folder, we have the papers we have used for the work. In the `plots` folder, we have relevant images of the results. In the `results` folder, we have saved our results along with those of all the participants in the competition. In the `scripts` folder, you can find all the necessary Python functions to execute the notebooks.
