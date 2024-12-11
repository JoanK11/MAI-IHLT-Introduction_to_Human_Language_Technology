import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import nltk
from scripts.preprocessing import replace_contractions, lemmatize

def preprocess_dataset(data):
    data['tokens_0'] = data['sentence_0'].apply(nltk.word_tokenize)
    data['tokens_1'] = data['sentence_1'].apply(nltk.word_tokenize)

    data['tokens_0'] = data['tokens_0'].apply(lambda tokens: [word for word in tokens if any(char.isalnum() for char in word)])
    data['tokens_1'] = data['tokens_1'].apply(lambda tokens: [word for word in tokens if any(char.isalnum() for char in word)])

    data['tokens_0'] = data['tokens_0'].apply(replace_contractions)
    data['tokens_1'] = data['tokens_1'].apply(replace_contractions)

    data['lemmas_0'] = data.apply(lambda row: lemmatize(row, "tokens_0", True), axis=1)
    data['lemmas_1'] = data.apply(lambda row: lemmatize(row, "tokens_1", True), axis=1)

    data['sentence_lemmas_0'] = data.apply(lambda row: " ".join(row["lemmas_0"]), axis=1)
    data['sentence_lemmas_1'] = data.apply(lambda row: " ".join(row["lemmas_1"]), axis=1)
    
    return data

def compute_best_results(best_model_name: str, best_model):
    datasets = ["MSRpar", "MSRvid", "SMTeuroparl"]#, "OnWN", "SMTnews"]

    results = {}
    normalized_results = {}

    # Load dataset names for filtering features
    test_data = pd.read_csv('datasets/test_preprocessed.csv')

    # Load test features
    features_test = pd.read_csv('features/features_test.csv')

    for dataset in datasets:
        # Load input sentences and gold scores for the dataset
        input_file = f"./datasets/test-gold/STS.input.{dataset}.txt"
        gold_scores_file = f"./datasets/test-gold/STS.gs.{dataset}.txt"

        inputs = pd.read_csv(input_file, delimiter='\t', header=None, names=['sentence_0', 'sentence_1'])
        gold_scores = pd.read_csv(gold_scores_file, delimiter='\t', header=None).values.flatten()

        # Preprocess the dataset
        inputs = preprocess_dataset(inputs)

        # Filter features based on dataset name
        relevant_features = features_test[test_data['dataset_name'] == dataset]

        # Predict scores using the model
        predictions = best_model.predict(relevant_features)

        # Calculate Pearson correlation
        pearson_corr, _ = pearsonr(predictions, gold_scores)

        # Normalize predictions and calculate normalized Pearson correlation (canviar codi!!)
        normalized_predictions = (predictions - predictions.mean()) / predictions.std()
        normalized_gold_scores = (gold_scores - gold_scores.mean()) / gold_scores.std()
        normalized_pearson_corr, _ = pearsonr(normalized_predictions, normalized_gold_scores)

        # Store results
        results[dataset] = pearson_corr
        normalized_results[dataset] = normalized_pearson_corr
    
    # Calculate overall statistics
    all_scores = list(results.values())
    all_normalized_scores = list(normalized_results.values())
    all_mean = sum(all_scores) / len(all_scores)

    return {
        "all": all_scores,
        "all_nrm": all_normalized_scores,
        "mean": all_mean,
        **results
    }
    

def calculate_and_compare_results(own_model_path, official_results_path, output_path):
    """
    Calculates the required metrics for the own model and compares them with official results.
    
    Parameters:
    - own_model_path: Path to the CSV file containing own model results.
    - official_results_path: Path to the CSV file containing official results.
    - output_path: Path to save the comparison results.
    
    The function assumes that the official_results.csv has the following columns:
    Run, ALL, Rank_ALL, ALLnrm, Rank_ALLnrm, Mean, Rank_Mean, MSRpar, MSRvid, SMT-eur, On-WN, SMT-news
    """
    # Load own model results
    own_results = pd.read_csv(own_model_path)
    
    # Extract the Mean Pearson CV Score
    own_mean_pearson = own_results.loc[0, 'Mean_Pearson_CV_Score']
    
    # Load official results
    official_results = pd.read_csv(official_results_path)
    
    # Append own model to official results
    own_run = {
        'Run': 'our-model',
        'ALL': own_mean_pearson,  # Assuming 'ALL' corresponds to overall Pearson
        # Assign placeholders for other metrics if not available
        'Rank_ALL': np.nan,
        'ALLnrm': np.nan,
        'Rank_ALLnrm': np.nan,
        'Mean': own_mean_pearson,  # Assuming 'Mean' corresponds to own mean
        'Rank_Mean': np.nan,
        'MSRpar': np.nan,
        'MSRvid': np.nan,
        'SMT-eur': np.nan,
        'On-WN': np.nan,
        'SMT-news': np.nan
    }
    
    # Append the own model run
    official_results = official_results.append(own_run, ignore_index=True)
    
    # Calculate ranks for 'ALL', 'ALLnrm', and 'Mean'
    official_results['Rank_ALL'] = official_results['ALL'].rank(method='min', ascending=False)
    official_results['Rank_ALLnrm'] = official_results['ALLnrm'].rank(method='min', ascending=False)
    official_results['Rank_Mean'] = official_results['Mean'].rank(method='min', ascending=False)
    
    # Save the updated official results with own model
    official_results.to_csv(output_path, index=False)
    print(f"Comparison results saved to {output_path}")

if __name__ == "__main__":
    # Define paths
    best_model_results_path = 'results/best_model_results.csv'
    official_results_path   = 'results/official_results.csv'
    comparison_output_path  = 'results/comparison_results.csv'
    
    # Calculate and compare results
    calculate_and_compare_results(best_model_results_path, official_results_path, comparison_output_path)