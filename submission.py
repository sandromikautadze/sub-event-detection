import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from utils import clean_tweets, get_cls_embeddings, process_and_merge_embeddings_by_similarity, EmbeddingDataset, ProjectionHead, SupervisedContrastiveLoss, load_and_process_tweets, calculate_top_tfidf_from_existing, load_embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import optuna
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#########################
#### DATA PROCESSING ####
#########################

reproduce_data_processing = False # set to True to run the entire pipeline (we suggest to keep it to False, as the entire process takes a lot of time)
if reproduce_data_processing is False:
    merged_df = pd.read_pickle("processed_data/contrastive_similarity_aggregated_processed2_no_teams_embeddings_with_labels.pkl")
    merged_df_test = pd.read_pickle("processed_data/contrastive_similarity_aggregated_processed2_no_teams_embeddings_with_labels_test.pkl")
    features = pd.read_csv("processed_data/features.csv")
    features_test = pd.read_csv("processed_data/val_features.csv")
else:
    # CLEANING
    li_train = []
    matchid_to_filename = {}
    for filename in tqdm(os.listdir("data/train_tweets"), desc="Processing Files"):
        df_temp = pd.read_csv(f"data/train_tweets/{filename}")
        df_temp = clean_tweets(df_temp, column_name="Tweet", replace_teams=True, remove_one_word_tweets=False)
        li_train.append(df_temp)
        unique_match_ids = df_temp['MatchID'].unique()
        for match_id in unique_match_ids:
            matchid_to_filename[match_id] = filename
    df_train = pd.concat(li_train, ignore_index=True)

    # TOKENIZATION
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    model = AutoModel.from_pretrained("vinai/bertweet-base")
    model = model.to(device)
    tokens = tokenizer(df_train["Tweet"].to_list(), max_length=50, padding="max_length", truncation=True, add_special_tokens=True, return_tensors="pt")

    # EMBEDDINGS AND 'CLS' EXTRACTION
    dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    final_cls_embeddings = get_cls_embeddings(model, dataloader, device)
    df_train['cls'] = list(final_cls_embeddings.numpy())

    # AGGREGATION BY SIMILARITY
    merged_df = process_and_merge_embeddings_by_similarity(df_train,
                                                        cls_column="cls", event_type_column="EventType", id_column="ID",
                                                        output_format="pkl",
                                                        output_path="processed_data/similarity_aggregated_processed2_no_teams_embeddings_with_labels.pkl")

    # CONTRASTIVE LEARNING (with already optimized hyperparameters)
    batch_size = 64
    full_dataset = EmbeddingDataset(merged_df)
    full_dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    projection_head_model = ProjectionHead(input_dim=768, hidden_dim=512, output_dim=256, dropout_p=0.2).to(device)
    contrastive_loss_function = SupervisedContrastiveLoss(high_temperature=1.0, low_temperature=0.2, total_epochs=50)
    optimizer = optim.Adam(projection_head_model.parameters(), lr=1e-4, weight_decay=1e-5)

    total_epochs = 50
    train_losses = []

    for current_epoch in range(total_epochs):
        contrastive_loss_function.set_epoch(current_epoch)
        projection_head_model.train()
        total_train_loss = 0

        for batch_embeddings, batch_labels in full_dataloader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)

            projected_embeddings = projection_head_model(batch_embeddings)
            batch_loss = contrastive_loss_function(projected_embeddings, batch_labels)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_train_loss += batch_loss.item()

        avg_train_loss = total_train_loss / len(full_dataloader)
        train_losses.append(avg_train_loss)

        print(f"Epoch [{current_epoch + 1}/{total_epochs}], Train Loss: {avg_train_loss:.4f}, "
            f"Temperature: {contrastive_loss_function.get_temperature():.4f}")

    projection_head_model.eval()
    projections = []
    with torch.no_grad():  
        for embedding in merged_df["aggregated_embedding"]:
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 768)
            projection = projection_head_model(embedding_tensor)  
            projections.append(projection.cpu().numpy().flatten())

    merged_df = merged_df.copy()
    merged_df["projections"] = projections
    merged_df = merged_df.drop(columns=['GameID', 'aggregated_embedding'])
    merged_df = merged_df.rename(columns={'projections': 'aggregated_embedding'})
    merged_df.to_pickle("processed_data/contrastive_similarity_aggregated_processed2_no_teams_embeddings_with_labels.pkl")

    # So, now the dataset has shape (2137, 3) with columns:
    # 'ID'
    # 'EventType'
    # 'aggregated_embedding' (256-dimensional)

    # We do all of this for the test set as well
    li_test = []
    matchid_to_filename_test = {}
    for filename in tqdm(os.listdir("data/eval_tweets"), desc="Processing Files"):
        df_temp = pd.read_csv(f"data/eval_tweets/{filename}")
        df_temp = clean_tweets(df_temp, column_name="Tweet", replace_teams=True, remove_one_word_tweets=True)
        li_test.append(df_temp)
        unique_match_ids_val = df_temp['MatchID'].unique()
        for match_id in unique_match_ids_val:
            matchid_to_filename_test[match_id] = filename
    df_test = pd.concat(li_test, ignore_index=True)

    tokens_test = tokenizer(df_test["Tweet"].to_list(), max_length=50, padding="max_length", truncation=True, add_special_tokens=True, return_tensors="pt")

    dataset_test = TensorDataset(tokens_test['input_ids'], tokens_test['attention_mask'])
    dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)
    final_cls_embeddings_test = get_cls_embeddings(model, dataloader_test, device)
    df_test['cls'] = list(final_cls_embeddings_test.numpy())

    merged_df_test = process_and_merge_embeddings_by_similarity(df_test,
                                                                cls_column="cls", event_type_column="MatchID", id_column="ID",
                                                                output_format="pkl",
                                                                output_path="processed_data/similarity_aggregated_processed2_no_teams_embeddings_with_labels_test.pkl")
    projection_head_model.eval()
    projections_test = []
    with torch.no_grad():  
        for embedding in merged_df_test["aggregated_embedding"]:
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 768)
            projection = projection_head_model(embedding_tensor)  
            projections_test.append(projection.cpu().numpy().flatten())

    merged_df_test = merged_df_test.copy()
    merged_df_test["projections"] = projections_test
    merged_df_test = merged_df_test.drop(columns=['MatchID', 'aggregated_embedding'])
    merged_df_test = merged_df_test.rename(columns={'projections': 'aggregated_embedding'})
    merged_df_test.to_pickle("processed_data/contrastive_similarity_aggregated_processed2_no_teams_embeddings_with_labels_test.pkl")
    
    # FEATURE ENGINEERING
    processed_tweets = load_and_process_tweets("data/train_tweets", )
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_features = tfidf_vectorizer.fit_transform(processed_tweets['Tweet']).toarray()
    tfidf_df = pd.DataFrame(tfidf_features, columns=tfidf_vectorizer.get_feature_names_out())
    processed_tweets = calculate_top_tfidf_from_existing(processed_tweets, tokens_column='Tokens', tfidf_df=tfidf_df, top_n=30)
    processed_tweets = processed_tweets.drop(columns=['Timestamp', 'Tweet','Tokens'])
    processed_tweets = processed_tweets.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()
    features = processed_tweets.copy()
    processed_tweets.to_csv('processed_data/features.csv', index=False)
    features = processed_tweets.copy()
    
    val_df = load_and_process_tweets('eval_tweets', 5)
    tfidf_features_val = tfidf_vectorizer.transform(val_df['Tweet']).toarray()
    tfidf_df_val = pd.DataFrame(tfidf_features, columns=tfidf_vectorizer.get_feature_names_out())
    val_df = calculate_top_tfidf_from_existing(val_df, tokens_column='Tokens', tfidf_df=tfidf_df, top_n=30)
    val_df = val_df.drop(columns=['Timestamp', 'Tweet','Tokens'])
    val_df = val_df.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()
    features_test = val_df.copy()
    val_df.to_csv('processed_data/val_features.csv', index=False)
    
###############
#### MODEL ####
###############

df = pd.read_csv("processed_data/features.csv")
all_games = df["MatchID"].unique()
np.random.seed(seed)
np.random.shuffle(all_games)
all_games = df["MatchID"].unique()
train_df = df[df["MatchID"].isin(all_games[:13])]
test_df = df[df["MatchID"].isin(all_games[13:])]

train_file = "processed_data/contrastive_similarity_aggregated_processed2_no_teams_embeddings_with_labels.pkl"
embeddings_train = load_embeddings(train_file)
embeddings_train["MatchID"] = embeddings_train["MatchID"].astype(int)

emb_train = embeddings_train[embeddings_train["MatchID"].isin(all_games[:13])]
emb_test = embeddings_train[embeddings_train["MatchID"].isin(all_games[13:])]

train_embeddings = np.vstack(emb_train['aggregated_embedding'].values)  # Shape: (num_train_samples, 256)
test_embeddings = np.vstack(emb_test['aggregated_embedding'].values)    # Shape: (num_test_samples, 256)

scaler = StandardScaler()
train_embeddings_scaled = scaler.fit_transform(train_embeddings)
test_embeddings_scaled = scaler.transform(test_embeddings)

pca_full = PCA().fit(train_embeddings_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components_80 = np.argmax(cumulative_variance >= 0.80) + 1

plot_pca_var_cumulative = True
if plot_pca_var_cumulative:
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
    plt.axhline(y=0.80, color='r', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.show()

# Apply PCA with 25 components or n_components_80, whichever is larger
desired_components = max(25, n_components_80)
pca = PCA(n_components=desired_components, random_state=42)
train_embeddings_pca = pca.fit_transform(train_embeddings_scaled)
test_embeddings_pca = pca.transform(test_embeddings_scaled)
print(f"PCA reduced embeddings to {desired_components} dimensions.")

# Create DataFrames for PCA components
train_pca_df = pd.DataFrame(train_embeddings_pca, columns=[f'PCA_{i+1}' for i in range(desired_components)])
test_pca_df = pd.DataFrame(test_embeddings_pca, columns=[f'PCA_{i+1}' for i in range(desired_components)])

# Add the ID column to the PCA DataFrames for merging
train_pca_df['ID'] = emb_train['ID'].values
test_pca_df['ID'] = emb_test['ID'].values

# Merge the PCA DataFrames with the original DataFrames using the ID column
train_merged_df = pd.merge(train_df, train_pca_df, on='ID')
test_merged_df = pd.merge(test_df, test_pca_df, on='ID')

# Remove the ID column from the merged DataFrames
train_merged_df.drop(columns=['ID'], inplace=True)
test_merged_df.drop(columns=['ID'], inplace=True)

# Assuming macro features are all columns except 'MatchID', 'PeriodID'
macro_columns = [col for col in train_merged_df.columns if col not in ['MatchID', 'PeriodID']]

train_features = train_merged_df[macro_columns]
test_features = test_merged_df[macro_columns]

print(train_features.shape)
print(test_features.shape)
print(train_features.head())

# Encode target
if 'EventType' not in train_features.columns:
    raise ValueError("EventType not found in the training features.")

y = train_features['EventType'].values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Drop EventType from training features
train_features = train_features.drop(columns=['EventType'])

print(f"Encoded classes: {le.classes_}")
print(f"Training features shape: {train_features.shape}")

# Use all games for cross-validation
unique_games = np.unique(all_games)
np.random.seed(42)
shuffled_games = np.random.permutation(unique_games)

n_splits = 5
folds = np.array_split(shuffled_games, n_splits)

X = train_features
y = y_encoded
game_ids = train_df['MatchID'].values

fold_indices = []
for fold_games in folds:
    val_mask = np.isin(game_ids, fold_games)
    val_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]
    fold_indices.append((train_idx, val_idx))

print("Created game-based fold indices using all games.")

# Focus on Random Forest only
model_params = {
    'random_forest': {
        'model': RandomForestClassifier(random_state=42),
        # Expanded hyperparameter space with no more than three values per hyperparameter
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    }
}

def tune_model_with_gridsearch(model_type, X, y, fold_indices, param_grid):
    """
    Uses GridSearchCV with the provided fold_indices to tune hyperparameters.
    """
    model_info = model_params[model_type]
    base_model = model_info['model']
    grid = model_info['params']

    cv_splits = [(train_idx, val_idx) for (train_idx, val_idx) in fold_indices]

    print(f"\nTuning {model_type} with GridSearchCV...")
    grid_search = GridSearchCV(estimator=base_model, param_grid=grid,
                               scoring='accuracy', cv=cv_splits, verbose=1, n_jobs=-1)
    grid_search.fit(X, y)
    print(f"Best parameters for {model_type}: {grid_search.best_params_}")
    return grid_search.best_estimator_, grid_search.best_params_

rf_best_model, rf_best_params = tune_model_with_gridsearch('random_forest', X, y, fold_indices, model_params['random_forest']['params'])

def evaluate_cv(model, X, y, fold_indices):
    accuracies = []
    for (train_idx, val_idx) in fold_indices:
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        cloned_model = clone(model)
        cloned_model.fit(X_train_fold, y_train_fold)
        preds = cloned_model.predict(X_val_fold)
        acc = accuracy_score(y_val_fold, preds)
        accuracies.append(acc)
    return np.mean(accuracies)

acc_rf = evaluate_cv(rf_best_model, X, y, fold_indices)
print("\nCross-Validation Accuracy for Random Forest (Best Params):")
print(f"Random Forest: {acc_rf:.4f}")

print(f"\nRetraining the best Random Forest model on the full dataset with best params: {rf_best_params}...")
rf_best_model.fit(X, y)
print("Retraining complete.")

test_file = "processed_data/contrastive_similarity_aggregated_processed2_no_teams_embeddings_with_labels_test.pkl"
final_test_df = pd.read_csv("processed_data/val_features.csv")
final_test_embeddings = load_embeddings(test_file)

print("Preparing final test embeddings...")
test_embeddings = np.vstack(final_test_embeddings['aggregated_embedding'].values)
test_embeddings_scaled = scaler.transform(test_embeddings)  # scaler fitted on training embeddings
test_embeddings_pca = pca.transform(test_embeddings_scaled)  # pca fitted on training embeddings
print(f"Test embeddings PCA shape: {test_embeddings_pca.shape}")

# Create DataFrame for PCA components in test set
pca_columns = [f'PCA_{i+1}' for i in range(pca.n_components_)]
test_pca_df = pd.DataFrame(test_embeddings_pca, columns=pca_columns)
test_pca_df['ID'] = final_test_embeddings['ID'].values

# Merge PCA embeddings with macro features in final_test_df
print("Merging PCA components with macro features in the final test set...")
final_merged_test_df = pd.merge(final_test_df, test_pca_df, on='ID', how='inner')

# Remove ID and any unnecessary columns (e.g., 'MatchID', 'PeriodID', 'EventType' if present)
columns_to_remove = ['ID', 'MatchID', 'PeriodID', 'EventType']
final_merged_test_df.drop(columns=[c for c in columns_to_remove if c in final_merged_test_df.columns], inplace=True, errors='ignore')

print("Final test features shape:", final_merged_test_df.shape)
# Predict on the final test set
print("Predicting on final test set...")
final_preds = rf_best_model.predict(final_merged_test_df)
final_preds_labels = le.inverse_transform(final_preds)

## SUBMISSION

submission = pd.DataFrame({
    "ID": final_test_df["ID"],
    "EventType": final_preds_labels
})
submission.to_csv("LAST_submission_EVER.csv", index=False)
print("Final submission 'submission.csv' created successfully.")

# feature importance

importances = rf_best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

print("Top 20 Features by Importance:")
print(feature_importance_df.head(20))
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['feature'].head(20)[::-1], 
         feature_importance_df['importance'].head(20)[::-1])
plt.xlabel('Feature Importance')
plt.ylabel('Feature Name')
plt.title('Top 20 Important Features in the Random Forest Model')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()