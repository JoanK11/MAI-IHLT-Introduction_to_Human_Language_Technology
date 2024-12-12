import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compare_results(df, best_results):
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

def plot_pearson_comparison(pearson_results, competition_averages, title="Model vs Competition Average Pearson Correlation"):
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