import os

def load_dataset(directory, dataset_category='train'):
    """
    Loads the dataset from the specified directory.

    Parameters:
        directory (str): Path to the datasets directory.
        dataset_category (str): Category of the dataset to load ('train' or 'test').

    Returns:
        list: A list of tuples containing sentence pairs, similarity scores, and dataset names.
    """
    file_inputs, file_gs = [], []

    # Determine the folder path based on the dataset category
    folder_path = os.path.join(directory, 'train' if dataset_category == 'train' else 'test-gold')

    # Collect input files
    file_inputs = [
        os.path.join(folder_path, fname)
        for fname in os.listdir(folder_path)
        if fname.startswith('STS.input') or fname.startswith('STS.input.surprise')
    ]

    # Collect gold standard files
    file_gs = [
        os.path.join(folder_path, fname)
        for fname in os.listdir(folder_path)
        if (fname.startswith('STS.gs') or fname.startswith('STS.gs.surprise')) and not fname.endswith('ALL.txt')
    ]

    combined_data = []

    # Process each pair of input and gold standard files
    for input_path, gs_path in zip(sorted(file_inputs), sorted(file_gs)):
        base_name = os.path.basename(input_path).split('.')[2]
        if base_name.startswith('surprise'):
            base_name = '.'.join(os.path.basename(input_path).split('.')[2:4])

        with open(input_path, 'r', encoding='utf-8') as inp_file, open(gs_path, 'r', encoding='utf-8') as gs_file:
            input_pairs = [line.strip().split('\t') for line in inp_file]
            gs_scores = [float(line.strip()) for line in gs_file]

            combined_data.extend(
                [
                    (pair[0], pair[1], score, base_name)
                    for pair, score in zip(input_pairs, gs_scores)
                ]
            )

    return combined_data