"""
Authors: Lara Hofman, Sandro Mikautadze, Elio Samaha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm


######################
##### FUNCTIONS ######
######################

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
    if criterion == "mean":
        return np.mean(np.array(embeddings), axis=0)
    
    
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

def train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=100, seed=42, use_tqdm=True):
    """
    Train and validate a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to train and validate.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): The loss function used to compute training and validation losses.
        optimizer (torch.optim.Optimizer): The optimizer used to update model parameters.
        device (torch.device): The device to perform computations on (e.g., 'cpu' or 'cuda').
        epochs (int, optional): The number of training epochs. Defaults to 100.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.
        use_tqdm (bool, optional): Whether to display progress bars using tqdm. Defaults to True.

    Returns:
        dict: A dictionary containing the training and validation losses and accuracies over all epochs. 
              Keys include:
                - 'train_loss': List of training losses for each epoch.
                - 'val_loss': List of validation losses for each epoch.
                - 'train_accuracy': List of training accuracies for each epoch.
                - 'val_accuracy': List of validation accuracies for each epoch.
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

        train_losses.append(running_loss / len(train_loader))
        train_accuracy = correct_preds / total_preds
        train_accuracies.append(train_accuracy)

        # VALIDATION
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

        val_losses.append(val_loss / len(val_loader))
        val_accuracy = correct_preds / total_preds
        val_accuracies.append(val_accuracy)

        print(f"Training Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_losses[-1]:.4f}, Accuracy: {val_accuracy:.4f}")

    print("Training Done")
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies
    }

    return history

# add here new functions if needed

######################
####### MODELS #######
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
    
# add here new models if needed
