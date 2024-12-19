import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compare_results_extended(df, best_results):
    """
    Process the input CSV file, add the new result, and recalculate ranks.
    
    Args:
        df (pd.Dataframe): Dataframe containing the CSV file with the official results.
        best_results (dict): Dictionary containing the new result to be added.
    
    Returns:
        pd.DataFrame: Processed DataFrame with new results and ranks.
    """
    # Convert the best results to a DataFrame
    new_row = pd.DataFrame([best_results])
    
    # Combine the existing DataFrame with the new row
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Function to calculate ranks for a specific column
    def calculate_ranks(column):
        # Sort the column in descending order (higher is better)
        return df[column].rank(method='dense', ascending=False).astype(int)
    
    # Calculate ranks for ALL, ALLnrm, and Mean
    df['Rank_ALL'] = calculate_ranks('ALL')
    df['Rank_ALLnrm'] = calculate_ranks('ALLnrm')
    df['Rank_Mean'] = calculate_ranks('Mean')
    
    # Calculate ranks for additional columns
    df['Rank_MSRpar'] = calculate_ranks('MSRpar')
    df['Rank_MSRvid'] = calculate_ranks('MSRvid')
    df['Rank_SMT-eur'] = calculate_ranks('SMT-eur')
    df['Rank_On-WN'] = calculate_ranks('On-WN')
    df['Rank_SMT-news'] = calculate_ranks('SMT-news')
    
    # Find rows with specific runs
    baseline_row = df[df['Run'] == '00-baseline/task6-baseline']
    our_model_row = df[df['Run'] == 'our_model']
    
    # Filter for rows where Rank_ALL is between 1 and 10
    top_10_df = df[df['Rank_ALL'].between(1, 10)]
    
    # Combine top 10 with baseline and our model rows
    # First, concatenate all rows
    combined_rows = pd.concat([
        top_10_df, 
        baseline_row, 
        our_model_row
    ])
    
    # Remove duplicates, keeping the first occurrence
    final_df = combined_rows.drop_duplicates(subset='Run', keep='first')
    
    # Sort by Rank_ALL to ensure consistent order
    final_df = final_df.sort_values('Rank_ALL')
    
    # Reset index and drop the index column to remove numerical index
    final_df = final_df.reset_index(drop=True)
    
    return final_df


def plot_pearson_comparison(pearson_results, competition_averages, title="Model vs Top 10 Competition Average Pearson Correlation"):
    """
    Plots a grouped bar chart comparing Pearson correlations for a model and competition averages.

    Parameters:
        pearson_results (dict): Model's Pearson correlation results (dataset_name: value).
        competition_averages (dict): Competition average Pearson correlations (dataset_name: value).
        title (str): Title of the plot.

    Returns:
        fig, ax: The matplotlib figure and axes objects for further customization or saving.
    """
    # Extract datasets and scores
    datasets = list(pearson_results.keys())
    model_scores = list(pearson_results.values())
    competition_scores = [competition_averages[ds] for ds in datasets]

    # Positions and width for the bars
    x = np.arange(len(datasets))
    width = 0.35

    # Get the tab10 color palette
    cmap = plt.get_cmap('tab10')
    model_color = cmap(0)        # First color
    competition_color = cmap(1)  # Second color

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, model_scores, width, label='Our Model', color=model_color)
    bars2 = ax.bar(x + width/2, competition_scores, width, label='Competition Avg', color=competition_color)

    # Add labels and title
    ax.set_ylabel('Pearson Correlation')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()

    # Add value labels on top of each bar
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    return fig, ax


def plot_pearson_comparison2(
    pearson_results_mixed, pearson_corr_mixed,
    pearson_results_syntactic, pearson_corr_syntactic,
    pearson_results_lexical, pearson_corr_lexical,
    title="Model Dimensions Comparison"
):
    """
    Plots a grouped bar chart comparing Pearson correlations for multiple models.
    
    Parameters:
        pearson_results_mixed (dict): Dictionary of dataset_name: Pearson correlation for the Mixed Model.
        pearson_corr_mixed (float): Mean Pearson correlation for the Mixed Model on the test set.
        pearson_results_syntactic (dict): Dictionary of dataset_name: Pearson correlation for the Syntactic Model.
        pearson_corr_syntactic (float): Mean Pearson correlation for the Syntactic Model on the test set.
        pearson_results_lexical (dict): Dictionary of dataset_name: Pearson correlation for the Lexical Model.
        pearson_corr_lexical (float): Mean Pearson correlation for the Lexical Model on the test set.
        title (str): Title of the plot.
    
    Returns:
        fig, ax: The matplotlib figure and axes objects for further customization or saving.
    """
    # Define models and their corresponding scores
    models = ['Mixed Model', 'Syntactic Model', 'Lexical Model']
    pearson_scores = {
        'Mixed Model': pearson_results_mixed,
        'Syntactic Model': pearson_results_syntactic,
        'Lexical Model': pearson_results_lexical
    }
    pearson_corr_test = {
        'Mixed Model': pearson_corr_mixed,
        'Syntactic Model': pearson_corr_syntactic,
        'Lexical Model': pearson_corr_lexical
    }
    
    # Extract dataset names
    dataset_names = list(pearson_results_mixed.keys())
    
    # Create groups: 'ALL' and each dataset
    groups = ['ALL'] + dataset_names
    n_groups = len(groups)
    
    # Number of models
    num_models = len(models)
    
    # Define bar width and positions
    width = 0.2
    x = np.arange(n_groups)
    
    # Calculate offsets for each model
    offsets = np.linspace(-width, width, num_models)
    
    # Get the tab10 color palette
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(num_models)]
    
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Plot 'ALL' group (test set mean Pearson correlations)
    for i, model in enumerate(models):
        score = pearson_corr_test[model]
        ax.bar(x[0] + offsets[i], score, width, label=model if i == 0 else "", color=colors[i])
        ax.annotate(f'{score:.2f}',
                    xy=(x[0] + offsets[i], score),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Plot each dataset group (per-dataset Pearson correlations)
    for idx, ds in enumerate(dataset_names):
        group_idx = idx + 1  # Offset by 1 because 'ALL' is the first group
        for i, model in enumerate(models):
            score = pearson_scores[model].get(ds, 0)
            ax.bar(x[group_idx] + offsets[i], score, width, label=model if (idx == 0 and i == 0) else "", color=colors[i])
            ax.annotate(f'{score:.2f}',
                        xy=(x[group_idx] + offsets[i], score),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    # Set labels and title
    ax.set_ylabel('Pearson Correlation')
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha='right', fontsize=12)
    
    # Create custom legend
    handles = [
        plt.Rectangle((0,0),1,1, color=colors[0]),
        plt.Rectangle((0,0),1,1, color=colors[1]),
        plt.Rectangle((0,0),1,1, color=colors[2])
    ]
    labels = models
    ax.legend(handles, labels, fontsize=12)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    return fig, ax