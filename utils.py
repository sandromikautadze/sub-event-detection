"""
Authors: Lara Hofman, Sandro Mikautadze, Elio Samaha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import os
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter


######################
##### FUNCTIONS ######
######################


def find_teams(data, teams_dict):
    """
    Identifies the two most commonly mentioned teams in a DataFrame column containing tweets.

    Args:
        data (pd.DataFrame): The DataFrame containing the tweets.
        teams_dict (dict): A dictionary mapping team full names to their acronyms.

    Returns:
        tuple: A tuple containing the full names of the two most commonly mentioned teams.

    Raises:
        ValueError: If there are not enough teams mentioned in the dataset to identify two distinct teams.
    """
    team_counter = Counter()

    for tweet in data['Tweet']:
        for team_full, team_acronym in teams_dict.items():
            # Combine counts of full names and acronyms
            if re.search(rf"\b{team_full}\b", tweet, re.IGNORECASE) or re.search(rf"\b{team_acronym}\b", tweet, re.IGNORECASE):
                team_counter[team_full] += 1  # Count everything under the full name

    # Identify the two most common teams
    most_common_teams = team_counter.most_common(2)
    if len(most_common_teams) < 2:
        raise ValueError("Not enough teams found in the dataset to identify team_1 and team_2.")

    # Return the full names of the two teams
    team_1 = most_common_teams[0][0]
    team_2 = most_common_teams[1][0]

    return team_1, team_2

def replace_team_mentions(text, team_1, team_2, teams_dict):
    """
    Replaces mentions of specific teams in a text with generic tags for normalization.

    Args:
        text (str): The input text in which team mentions should be replaced.
        team_1 (str): The full name of the first team to normalize mentions as "<team_1>".
        team_2 (str): The full name of the second team to normalize mentions as "<team_2>".
        teams_dict (dict): A dictionary mapping team full names to their acronyms.

    Returns:
        str: The text with mentions of team_1 replaced by "<team_1>", mentions of team_2 replaced by "<team_2>", 
             and mentions of other teams replaced by "<team>".
    """
    # Replace mentions of team_1
    for alias in [team_1, teams_dict[team_1]]:
        text = re.sub(rf"\b{alias}\b", "<team_1>", text, flags=re.IGNORECASE)

    # Replace mentions of team_2
    for alias in [team_2, teams_dict[team_2]]:
        text = re.sub(rf"\b{alias}\b", "<team_2>", text, flags=re.IGNORECASE)

    # Replace mentions of all other teams
    for team_full, team_acronym in teams_dict.items():
        if team_full != team_1 and team_full != team_2:
            text = re.sub(rf"\b{team_full}\b", "<team>", text, flags=re.IGNORECASE)
            text = re.sub(rf"\b{team_acronym}\b", "<team>", text, flags=re.IGNORECASE)

    return text

def clean_tweets(df, column_name="Tweet", replace_teams=False, teams_dict = None, remove_one_word_tweets=False):
    """
    Cleans a DataFrame column containing tweets by performing various preprocessing steps.

    Args:
        df (pd.DataFrame): The DataFrame containing the tweets to clean.
        column_name (str, optional): The name of the column to clean. Defaults to "Tweet".
        replace_teams (bool, optional): Whether to replace team mentions with generic tags. Defaults to False.
        teams_dict (dict, optional): A dictionary mapping team names to their acronyms. Defaults to a predefined dictionary.
        remove_one_word_tweets (bool, optional): Whether to remove tweets with only one word. Defaults to False.

    Returns:
        pd.DataFrame: The DataFrame with the specified column cleaned.
    """
    # # Emoji regex pattern
    # EMOJI_PATTERN = re.compile(
    #     "[" 
    #     "\U0001F600-\U0001F64F"  # Emoticons
    #     "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
    #     "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
    #     "\U0001F1E0-\U0001F1FF"  # Flags
    #     "\U00002700-\U000027BF"  # Dingbats
    #     "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    #     "\U00002600-\U000026FF"  # Miscellaneous Symbols
    #     "\U00002B50-\U00002B55"  # Miscellaneous Symbols and Arrows
    #     "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    #     "]+",
    #     flags=re.UNICODE,
    # )
    
    # We do this to make it adaptable to other datasets eventually
    if teams_dict is None:
        teams_dict = {
        "South Africa": "RSA", "Argentina": "ARG", "Australia": "AUS", "Brazil": "BRA",
        "Cameroon": "CMR", "Chile": "CHI", "Costa Rica": "CRC", "Denmark": "DEN",
        "England": "ENG", "France": "FRA", "Germany": "GER", "Ghana": "GHA",
        "Honduras": "HON", "Italy": "ITA", "Ivory Coast": "CIV", "Japan": "JPN",
        "Mexico": "MEX", "Netherlands": "NED", "New Zealand": "NZL", "Nigeria": "NGA",
        "North Korea": "PRK", "Paraguay": "PAR", "Portugal": "POR", "Slovakia": "SVK",
        "Slovenia": "SLO", "South Korea": "KOR", "Spain": "ESP", "Switzerland": "SUI",
        "United States": "USA", "Uruguay": "URU", "Algeria": "ALG", "Serbia": "SRB",
        "Belgium": "BEL", "Bosnia and Herzegovina": "BIH", "Colombia": "COL",
        "Croatia": "CRO", "Ecuador": "ECU", "Greece": "GRE", "Iran": "IRN",
        "Russia": "RUS"
        }

    df = df.drop_duplicates(subset=[column_name], keep="first") # Drop duplicates
    df = df[~df[column_name].str.startswith('RT')] # Remove retweets
    df.loc[:, column_name] = (
        df[column_name]
        .str.lower() # lowercase
        # .apply(lambda text: re.sub(r'@\w+', '', text)) # Remove mentions
        .apply(lambda text: re.sub(r'#(\w+)', r'\1', text)) # Remove hashtag symbol
        # .apply(lambda text: re.sub(r'http[s]?\S+', '', text)) # Remove URLs
        # .apply(lambda text: re.sub(EMOJI_PATTERN, '', text)) # Remove emojis
        .apply(lambda text: text.replace('\n', '')) # Remove \n
        # .apply(lambda text: re.sub(r'[^\w\s!?]', '', text)) # Remove punctuation except !,?
        .apply(lambda text: re.sub(r'\s+', ' ', text).strip()) # Remove extra spaces and trim
    )
    
    # Replace team mentions with generic tags
    if replace_teams:
        team_1, team_2 = find_teams(df, teams_dict)
        df.loc[:, column_name] = df[column_name].apply(lambda text: replace_team_mentions(text, team_1, team_2, teams_dict))

    df = df[df[column_name] != ''] # Remove rows with empty tweets
    df = df[df[column_name].apply(lambda text: bool(re.search(r'[a-zA-Z0-9]', text)))] # Remove rows with primarily non-ASCII characters
    
    # remove tweets with one word
    if remove_one_word_tweets:
        df = df[df[column_name].apply(lambda text: len(text.split()) > 1)] 

    return df

def get_length_info(series, tokenizer, percentiles=[50, 90, 95, 99], plot=True, verbose=True):
    """
    Analyze and visualize the tokenized length of text sequences in a series.

    Args:
        series (pd.Series): A Pandas Series containing text data.
        tokenizer (Tokenizer): A tokenizer with an `encode` method (e.g., HuggingFace tokenizer).
        percentiles (list of int, optional): Percentiles to calculate for the token lengths. Defaults to [50, 90, 95, 99].
        plot (bool, optional): Whether to plot a histogram of the token lengths. Defaults to True.
        verbose (bool, optional): Whether to print statistical summaries. Defaults to True.

    Returns:
        list: A list of token lengths for each text in the series.
    """
    lengths = [len(tokenizer.encode(sent, add_special_tokens=True)) for sent in series]
    lengths_np = np.array(lengths)

    if verbose:
        print("Token Length Statistics:")
        print(f"  Max length: {np.max(lengths_np)}")
        print(f"  Average length: {np.mean(lengths_np):.2f}")
        for percentile in percentiles:
            print(f"  {percentile}th percentile length: {np.percentile(lengths_np, percentile):.2f}")

    if plot:
        plt.figure(figsize=(10, 6))
        plt.hist(lengths_np, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(lengths_np), color='red', linestyle='--', label=f'Mean ({np.mean(lengths_np):.2f})')
        plt.axvline(np.percentile(lengths_np, 95), color='green', linestyle='--', label=f'95th Percentile ({np.percentile(lengths_np, 95):.2f})')
        plt.title("Distribution of Token Lengths")
        plt.xlabel("Token Length")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

def get_cls_embeddings(model, dataloader, device):
    """
    Extracts [CLS] token embeddings for each input sequence in the provided dataloader.

    Args:
        model (torch.nn.Module): The model used to obtain the embedding.
        dataloader (torch.utils.data.DataLoader): DataLoader providing batches of tokenized input data.
            Each batch should be a tuple containing:
            - input_ids (torch.Tensor): Tensor of token IDs with shape (batch_size, sequence_length).
            - attention_mask (torch.Tensor): Tensor indicating which tokens are padding with shape (batch_size, sequence_length).
        device (torch.device): The device (CPU or GPU) on which computations will be performed.

    Returns:
        torch.Tensor: A tensor containing the [CLS] token embeddings for all input sequences of shape (total_samples, hidden_size)
    """
    cls_embeddings_list = []

    for batch in tqdm(dataloader):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
        
        with torch.no_grad():
            token_embeddings = model(**tokens)
        
        cls_embeddings = token_embeddings.last_hidden_state[:, 0, :]
        cls_embeddings_list.append(cls_embeddings.cpu())
        
    print("Done!")
    
    return torch.cat(cls_embeddings_list, dim=0)


def aggregate_embeddings(embeddings, criterion = "mean"):
    """
    Helper that aggregates the embeddings according to a specific criterion.

    Args:
        embeddings (torch.Tensor): Embeddings to aggregate.
        criterion (str, optional): Type of aggregation. Defaults to "mean".

    Returns:
        np.array: The aggregated embedding.
    """
    # other criterions to be added.
    embeddings = np.array(embeddings)
    if criterion == "mean":
        return np.mean(embeddings, axis=0)    
    
    
def process_and_merge_embeddings(df, cls_column, id_column, event_column, aggregation_func, criterion="mean", output_format=None, output_path=None):
    """
    Aggregates embeddings for each unique ID and merges them with their associated labels.

    Args:
        df (pd.DataFrame): The input DataFrame containing embeddings and labels.
        cls_column (str): The name of the column containing individual embeddings (e.g., 'cls').
        id_column (str): The name of the column used for grouping (e.g., 'ID').
        event_column (str): The name of the column containing labels (e.g., 'EventType').
        aggregation_func (callable): A function to aggregate embeddings (e.g., np.mean, np.sum).
            The function must accept a series of embeddings and an optional `criterion` parameter.
        criterion (str, optional): A parameter to specify the aggregation method within `aggregation_func`.
            Defaults to "mean".
        output_format (str, optional): The format to save the output ('csv' or 'pkl'). Defaults to None.
        output_path (str, optional): The path to save the output file. Required if `output_format` is specified.

    Returns:
        tuple:
            - pd.DataFrame: A DataFrame containing aggregated embeddings for each unique ID.
            - pd.DataFrame: A merged DataFrame containing aggregated embeddings and associated labels.

    Raises:
        ValueError: If `output_path` is not provided when `output_format` is specified.
        ValueError: If an invalid `output_format` is provided.
    """
    
    # aggregate embeddings
    aggregated_embeddings = (
        df.groupby(id_column)[cls_column]
        .apply(lambda x: aggregation_func(x, criterion=criterion))
        .reset_index()
        .rename(columns={cls_column: 'aggregated_embedding'})
    )
    
    # merge with labels
    merged_df = pd.merge(
        aggregated_embeddings,
        df[[id_column, event_column]].drop_duplicates(),
        on=id_column,
        how='inner'
    )
    
    # save output if requested
    if output_format:
        if not output_path:
            raise ValueError("output_path must be specified if output_format is provided.")
        
        if output_format == 'csv':
            print(f"Saving merged DataFrame as {output_format} in {output_path}")
            merged_df.to_csv(output_path, index=False)
            print("Saved!")
        elif output_format == 'pkl':
            print(f"Saving merged DataFrame as {output_format} in {output_path}")
            merged_df.to_pickle(output_path)
            print("Saved!")
        else:
            raise ValueError("Invalid output_format. Choose 'csv' or 'pkl'.")
        
    return aggregated_embeddings, merged_df

def process_and_merge_embeddings_by_similarity(df, cls_column="cls", event_type_column="EventType", id_column="ID", output_format=None, output_path=None):
    """
    Aggregates embeddings per ID and merges the result with unique event-related data.

    Args:
        df (pd.DataFrame): The input DataFrame containing embeddings and labels.
        cls_column (str): The name of the column containing individual embeddings (e.g., 'cls').
        event_type_column (str): The name of the column containing labels (e.g., 'EventType').
        id_column (str): The name of the column used for grouping (e.g., 'ID').
        output_format (str, optional): The format to save the output ('csv' or 'pkl'). Defaults to None.
        output_path (str, optional): The path to save the output file. Required if `output_format` is specified.

    Returns:
        pd.DataFrame: A merged DataFrame containing aggregated embeddings and associated labels.
        
    Raises:
        ValueError: If `output_path` is not provided when `output_format` is specified.
        ValueError: If an invalid `output_format` is provided.
    """
    id_aggregated_embeddings = []

    # Group by ID
    id_groups = df.groupby(id_column)

    for unique_id, id_samples in id_groups:
        # Group the ID data by EventType
        event_type_groups = id_samples.groupby(event_type_column)
        event_type_aggregations = []
        total_tweets_in_id = len(id_samples)

        for event_type, event_type_samples in event_type_groups:
            # Extract embeddings for the current event type
            embeddings = np.vstack(event_type_samples[cls_column])  # (N, D)

            # Compute the reference embedding (mean)
            reference_embedding = np.mean(embeddings, axis=0)

            # Compute cosine similarities with the reference embedding
            similarities = cosine_similarity(embeddings, reference_embedding.reshape(1, -1)).flatten()

            # Normalize similarities using softmax to get attention weights
            attention_weights = torch.softmax(torch.tensor(similarities), dim=0).numpy()

            # Compute weighted aggregation for this event type
            weighted_aggregation = np.dot(attention_weights, embeddings)

            # Store the weighted aggregation with its contribution weight (based on size)
            event_type_aggregations.append((weighted_aggregation, len(embeddings)))

        # Combine weighted aggregations from all event types
        final_id_embedding = sum(
            (event_size / total_tweets_in_id) * event_embedding
            for event_embedding, event_size in event_type_aggregations
        )
        id_aggregated_embeddings.append({"ID": unique_id, "aggregated_embedding": final_id_embedding})

    # Convert to DataFrame
    aggregated_df = pd.DataFrame(id_aggregated_embeddings)

    # Merge with labels or unique event-related data
    merged_df = pd.merge(
        aggregated_df,
        df[[id_column, event_type_column]].drop_duplicates(),
        on=id_column,
        how="inner"
    )
    
    # save output if requested
    if output_format:
        if not output_path:
            raise ValueError("output_path must be specified if output_format is provided.")
        
        if output_format == 'csv':
            print(f"Saving merged DataFrame as {output_format} in {output_path}")
            merged_df.to_csv(output_path, index=False)
            print("Saved!")
        elif output_format == 'pkl':
            print(f"Saving merged DataFrame as {output_format} in {output_path}")
            merged_df.to_pickle(output_path)
            print("Saved!")
        else:
            raise ValueError("Invalid output_format. Choose 'csv' or 'pkl'.")
    
    return merged_df


def train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=100, seed=42, use_tqdm=True):
    """
    Train and (optionally) validate a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader or None): DataLoader for the validation dataset. If None, validation is skipped.
        criterion (torch.nn.Module): The loss function used to compute training (and validation) losses.
        optimizer (torch.optim.Optimizer): The optimizer used to update model parameters.
        device (torch.device): The device to perform computations on (e.g., 'cpu' or 'cuda').
        epochs (int, optional): The number of training epochs. Defaults to 100.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.
        use_tqdm (bool, optional): Whether to display progress bars using tqdm. Defaults to True.

    Returns:
        dict: A dictionary containing the training losses and accuracies over all epochs, and optionally validation losses and accuracies. 
              Keys include:
                - 'train_loss': List of training losses for each epoch.
                - 'train_accuracy': List of training accuracies for each epoch.
                - 'val_loss': List of validation losses for each epoch (empty if validation is skipped).
                - 'val_accuracy': List of validation accuracies for each epoch (empty if validation is skipped).
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}\n{'=' * 30}")

        # TRAINING
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        train_iterator = tqdm(train_loader, desc="Training", leave=True) if use_tqdm else train_loader

        for batch in train_iterator:
            X_batch, y_batch = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            logits = model(X_batch).squeeze()
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct = (preds == y_batch).sum().item()
            batch_accuracy = correct / y_batch.size(0)

            correct_preds += correct
            total_preds += y_batch.size(0)

            if use_tqdm:
                train_iterator.set_postfix(batch_loss=loss.item(), batch_accuracy=batch_accuracy)

        train_loss_epoch = running_loss / len(train_loader)
        train_losses.append(train_loss_epoch)
        train_accuracy = correct_preds / total_preds
        train_accuracies.append(train_accuracy)

        print(f"Training Loss: {train_loss_epoch:.4f}, Accuracy: {train_accuracy:.4f}")

        # VALIDATION (only if val_loader is provided)
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            correct_preds = 0
            total_preds = 0

            val_iterator = tqdm(val_loader, desc="Validating", leave=True) if use_tqdm else val_loader

            with torch.no_grad():
                for batch in val_iterator:
                    X_batch, y_batch = batch[0].to(device), batch[1].to(device)

                    logits = model(X_batch).squeeze()
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item()
                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    correct = (preds == y_batch).sum().item()
                    batch_accuracy = correct / y_batch.size(0)

                    correct_preds += correct
                    total_preds += y_batch.size(0)

                    if use_tqdm:
                        val_iterator.set_postfix(batch_loss=loss.item(), batch_accuracy=batch_accuracy)

            val_loss_epoch = val_loss / len(val_loader)
            val_losses.append(val_loss_epoch)
            val_accuracy = correct_preds / total_preds
            val_accuracies.append(val_accuracy)

            print(f"Validation Loss: {val_loss_epoch:.4f}, Accuracy: {val_accuracy:.4f}")
        else:
            # If no validation, append None or skip printing
            val_losses.append(None)
            val_accuracies.append(None)

    print("Training Done")
    history = {
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies
    }

    return history


## feature engineering
    
def find_teams(data, teams_dict):
    team_counter = Counter()

    for tweet in data['Tweet']:
        for team_full, team_acronym in teams_dict.items():
            # Combine counts of full names and acronyms
            if re.search(rf"\b{team_full}\b", tweet, re.IGNORECASE) or re.search(rf"\b{team_acronym}\b", tweet, re.IGNORECASE):
                team_counter[team_full] += 1  # Count everything under the full name

    # Identify the two most common teams
    most_common_teams = team_counter.most_common(2)
    if len(most_common_teams) < 2:
        raise ValueError("Not enough teams found in the dataset to identify team_1 and team_2.")

    # Return the full names of the two teams
    team_1 = most_common_teams[0][0]
    team_2 = most_common_teams[1][0]


    return team_1, team_2

def replace_team_mentions(text, team_1, team_2, teams_dict):
    # Replace mentions of team_1
    for alias in [team_1, teams_dict[team_1]]:
        text = re.sub(rf"\b{alias}\b", "team_1", text, flags=re.IGNORECASE)
        #text = re.sub(rf"#({alias})\b", "#team_1", text, flags=re.IGNORECASE)

    # Replace mentions of team_2
    for alias in [team_2, teams_dict[team_2]]:
        text = re.sub(rf"\b{alias}\b", "team_2", text, flags=re.IGNORECASE)
        #text = re.sub(rf"#({alias})\b", "#team_2", text, flags=re.IGNORECASE)

    # Replace mentions of all other teams
    for team_full, team_acronym in teams_dict.items():
        if team_full != team_1 and team_full != team_2:
            text = re.sub(rf"\b{team_full}\b", "team", text, flags=re.IGNORECASE)
            text = re.sub(rf"\b{team_acronym}\b", "team", text, flags=re.IGNORECASE)
            #text = re.sub(rf"#({team_full}|{team_acronym})\b", "#team", text, flags=re.IGNORECASE)

    return text

def preprocess_text(text):
    # Lowercase
    text = text.lower()

    # Replace URLs and mentions
    text = re.sub(r"http\S+|www\S+", "", text)  # Replace URLs with <URL>
    text = re.sub(r"@\w+", "", text)  # Replace mentions with <MENTION>

    # Remove emojis and special characters
    text = re.sub(r"[^\w\s]", "", text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenization
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
   # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

teams_dict = {
    "South Africa": "RSA", "Argentina": "ARG", "Australia": "AUS", "Brazil": "BRA",
    "Cameroon": "CMR", "Chile": "CHI", "Costa Rica": "CRC", "Denmark": "DEN",
    "England": "ENG", "France": "FRA", "Germany": "GER", "Ghana": "GHA",
    "Honduras": "HON", "Italy": "ITA", "Ivory Coast": "CIV", "Japan": "JPN",
    "Mexico": "MEX", "Netherlands": "NED", "New Zealand": "NZL", "Nigeria": "NGA",
    "North Korea": "PRK", "Paraguay": "PAR", "Portugal": "POR", "Slovakia": "SVK",
    "Slovenia": "SLO", "South Korea": "KOR", "Spain": "ESP", "Switzerland": "SUI",
    "United States": "USA", "Uruguay": "URU", "Algeria": "ALG", "Serbia": "SRB",
    "Belgium": "BEL", "Bosnia and Herzegovina": "BIH", "Colombia": "COL",
    "Croatia": "CRO", "Ecuador": "ECU", "Greece": "GRE", "Iran": "IRN",
    "Russia": "RUS"
}

team_performance_scores = {
    "Argentina": 95,
    "Australia": 45,
    "Brazil": 95,
    "Cameroon": 40,
    "Chile": 65,
    "Costa Rica": 42,
    "Denmark": 65,
    "England": 90,
    "France": 90,
    "Germany": 100,
    "Ghana": 60,
    "Honduras": 35,
    "Italy": 85,
    "Ivory Coast": 62,
    "Japan": 60,
    "Mexico": 65,
    "Netherlands": 85,
    "New Zealand": 30,
    "Nigeria": 60,
    "North Korea": 20,
    "Paraguay": 40,
    "Portugal": 80,
    "Slovakia": 58,
    "Slovenia": 42,
    "South Africa": 55,
    "South Korea": 60,
    "Spain": 100,
    "Switzerland": 75,
    "United States": 63,
    "Uruguay": 80,
    "Algeria": 55,
    "Serbia": 58,
    "Belgium": 90,
    "Bosnia and Herzegovina": 45,
    "Colombia": 78,
    "Croatia": 77,
    "Ecuador": 43,
    "Greece": 60,
    "Iran": 35,
    "Russia": 65
}


def extract_features(text):

    # Extract Hashtag count 
    hashtags = re.findall(r"#(\w+)", text)  
    hashtag_count = len(hashtags)

    # Count Exclamation Marks
    exclamation_count = text.count("!")

    return hashtag_count, exclamation_count

def load_and_process_tweets(
    directory, teams_dict=teams_dict, team_performance_scores=team_performance_scores, window_size=5
):
    """
    Load and process tweets from CSV files, extracting features, handling rolling statistics,
    and adding team performance scores.

    Args:
        directory (str): Path to the directory containing CSV files.
        teams_dict (dict): Mapping of team names to abbreviations.
        team_performance_scores (dict): Mapping of team names to performance scores.
        window_size (int): Window size of temporal features.

    Returns:
        pd.DataFrame: Processed DataFrame with extracted features.
    """

    # List to store processed data
    processed_data = []

    # Iterate over CSV files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # Load CSV file
            df = pd.read_csv(os.path.join(directory, filename))

            # Remove retweets and duplicates
            df = df[~df['Tweet'].str.startswith('RT')]
            df = df.drop_duplicates(subset=['Tweet'], keep="first").reset_index(drop=True)

            # Extract basic features
            df[['Hashtags Count', 'Exclamation_Count']] = df['Tweet'].apply(
                lambda x: pd.Series(extract_features(x))
            )

            # Preprocess tweets
            df['Tweet'] = df['Tweet'].apply(preprocess_text)
            df['TweetLength'] = df['Tweet'].apply(len)
            df['Tokens'] = df['Tweet'].apply(lambda x: x.split())

            # Calculate tweet count per period
            df['TweetCount_PerPeriod'] = df.groupby('PeriodID')['Tweet'].transform('count')

            # Find teams and calculate team performance scores
            team_1, team_2 = find_teams(df, teams_dict)
            print(f"Team 1: {team_1}, Team 2: {team_2}")
            print(f"Team 1 Score: {team_performance_scores[team_1]}, Team 2 Score: {team_performance_scores[team_2]}")

            # Replace team mentions in tweets
            df['Tweet'] = df['Tweet'].apply(lambda x: replace_team_mentions(x, team_1, team_2, teams_dict))

            # Calculate team score difference
            df['Team_Score'] = abs(team_performance_scores[team_1] - team_performance_scores[team_2])

            # Function to calculate previous period tweet frequencies
            def calculate_previous_period_features(group):
                group = group.sort_values('PeriodID')
                rolling_features = group[['PeriodID', 'TweetCount_PerPeriod']].drop_duplicates()
                rolling_features = rolling_features.sort_values('PeriodID')

                for i in range(1, 6):  # Calculate features for up to 5 previous periods
                    rolling_features[f'Previous{i}_TweetFreq'] = rolling_features['TweetCount_PerPeriod'].shift(i)

                group = group.merge(
                    rolling_features[['PeriodID'] + [f'Previous{i}_TweetFreq' for i in range(1, 6)]],
                    how='left',
                    on='PeriodID'
                )

                return group

            # Apply feature calculation for each match
            df = df.groupby('MatchID').apply(calculate_previous_period_features).reset_index(drop=True)

            # Replace NaN values with 0
            df.fillna(0, inplace=True)

            # Append processed DataFrame to the list
            processed_data.append(df)

    # Concatenate all processed DataFrames
    final_df = pd.concat(processed_data, ignore_index=True)

    return final_df

def calculate_top_tfidf_from_existing(df, tokens_column, tfidf_df, top_n=30):
    # Step 1: Identify the top-n TF-IDF words
    tfidf_scores = tfidf_df.sum(axis=0)  # Sum scores for each word
    top_tfidf_indices = tfidf_scores.argsort()[-top_n:][::-1]  # Indices of top-n words
    top_tfidf_words = tfidf_df.columns[top_tfidf_indices]  # Get the corresponding words
    
    
    # Step 2: Count the occurrences of top TF-IDF words in each row's tokens
    def count_top_tfidf_tokens(tokens):
        return sum(1 for token in tokens if token.lower() in top_tfidf_words)
    
    # Create a new column with the count
    df['TopTFIDFWordCount'] = df[tokens_column].apply(
        lambda tokens: count_top_tfidf_tokens(tokens) if isinstance(tokens, list) else 0
    )
    return df


# models

def load_embeddings(file_path):
    print(f"Loading embeddings from {file_path}...")
    df = pd.read_pickle(file_path)
    df["MatchID"] = df["ID"].apply(lambda x: x.split("_")[0])  # Extract MatchID from ID
    print(f"Loaded {len(df)} PeriodIDs.")
    return df

######################
####### CLASSES #######
######################            

class BertweetBaseMLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) model for binary classification tasks, designed to process 
    embeddings and predict a single output value (e.g., probability of a class).

    Args:
        embedding_dim (int): The dimensionality of the input embeddings.

    Attributes:
        network (torch.nn.Sequential): A sequential neural network comprising:
            - Fully connected layer (embedding_dim -> 512) with PReLU activation.
            - Fully connected layer (512 -> 256) with BatchNorm1d, PReLU activation, and Dropout(0.5).
            - Fully connected layer (256 -> 128) with BatchNorm1d, PReLU activation, and Dropout(0.5).
            - Fully connected layer (128 -> 64) with BatchNorm1d, PReLU activation, and Dropout(0.5).
            - Final fully connected layer (64 -> 1) for binary classification output.

    Methods:
        forward(x):
            Performs a forward pass through the network.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, embedding_dim).

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, 1), representing the raw logits.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.PReLU(),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(64, 1),
        )
        
    def forward(self, x):
        return self.network(x)

class BertweetMLP128Layer(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) model for dimensionality reduction and binary classification.

    The model reduces the input dimension (768) to 128 and maintains intermediate layers of size 128,
    before further reducing to 64 and finally outputting a single value.

    Args:
        embedding_dim (int): The dimensionality of the input embeddings.

    Attributes:
        network (torch.nn.Sequential): A sequential neural network comprising:
            - Fully connected layers (768 -> 128, 128 -> 128, ..., 128 -> 64 -> 1)
            - BatchNorm1d, PReLU activation, and Dropout applied at each layer.

    Methods:
        forward(x):
            Performs a forward pass through the network.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, embedding_dim).

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, 1), representing the raw logits.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)

class EmbeddingDataset(Dataset):
    def __init__(self, df):
        self.embeddings = np.stack(df["aggregated_embedding"].values).astype(np.float32) # 2136,768
        self.labels = df["EventType"].values.astype(np.int64)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=256, dropout_p=0.2):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),  
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, high_temperature=0.8, low_temperature=0.2, total_epochs=10, fraction_epochs=0.3):
        super(SupervisedContrastiveLoss, self).__init__()
        self.high_temperature = high_temperature
        self.low_temperature = low_temperature
        self.total_epochs = total_epochs
        self.fraction_epochs = fraction_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def get_temperature(self):
        # fraction of completed epochs
        fraction = self.current_epoch / float(self.total_epochs)

        if fraction < self.fraction_epochs:
            return self.high_temperature
        else:
            fraction_after_knee = (fraction - self.fraction_epochs) / (1.0 - self.fraction_epochs)
            decayed_temp = self.high_temperature - (self.high_temperature - self.low_temperature) * fraction_after_knee
            return max(decayed_temp, self.low_temperature)

    def forward(self, embeddings, labels):
    
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        temperature = self.get_temperature()
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
        
        # Mask to identify positive pairs
        labels = labels.unsqueeze(1)  # (batch_size, 1)
        mask = torch.eq(labels, labels.T).float()  # (batch_size, batch_size)
        
        exp_similarity = torch.exp(similarity_matrix)
        positive_pairs = mask * exp_similarity
        loss = -torch.log((positive_pairs.sum(dim=1) + 1e-8) / exp_similarity.sum(dim=1))
        return loss.mean()
    