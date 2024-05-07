import torch
import numpy as np
import dgl
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score
import pickle
import sys
import panel as pn
pn.extension()
sys.path.append('../..')
from pyeug import eug
sys.path.remove('../..')

def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2*(features - min_values).div(max_values-min_values) - 1


def train(g, features, labels, masks, model, epochs=2000, patience=1500, save_path='best_model.pth'):
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    best_val_acc = 0
    patience_counter = 0

    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        logits = model(g, features)
        # print(logits.shape, labels.shape, train_mask.shape)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Evaluate on validation set
        val_acc, val_f1 = evaluate(g, features, labels, val_mask, model)
        if epoch % 100 == 0:
            print(f"Epoch {epoch:05d} | Loss {loss.item():.4f} | Val Accuracy {val_acc:.4f} | Val F1 {val_f1:.4f}")

        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if patience_counter == patience:
            print("Early stopping triggered.")
            break


def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        predictions = torch.argmax(logits[mask], dim=1)
        acc = (predictions == labels[mask]).float().mean()
        f1 = f1_score(labels[mask].cpu(), predictions.cpu(), average='weighted')
    return acc.item(), f1


def preprocess_nba(nba):
    adj, features, idx_train, idx_val, idx_test, labels, sens, feat_names, sens_names \
        = nba.adj(), nba.features(), nba.idx_train(), nba.idx_val(), \
          nba.idx_test(), nba.labels(), nba.sens(), nba.feat_names(), nba.sens_names()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src, dst = adj.coalesce().indices()

    # Create the heterograph
    g = dgl.heterograph({('node', 'edge', 'node'): (src.cpu().numpy(), dst.cpu().numpy())})
    g = g.int().to(device)

    # convert idx_train, idx_val, idx_test to boolean masks
    train_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
    val_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    masks = train_mask, val_mask, test_mask

    features = features.to(device)
    labels = labels.to(device)

    # if sens is a tensor, convert to numpy array
    if isinstance(sens, torch.Tensor):
        sens = sens.cpu().numpy()
    # create a np array country with "US" for 1 and "oversea" for 0 in sens
    country = np.array(["US" if s == 1 else "Oversea" for s in sens])
    # get the index of "AGE" in feat_names
    age_idx = feat_names.index("AGE")
    feat_names.remove("AGE")
    age = features[:, age_idx].cpu().numpy()
    # cut age into '<25', '25-30', '>=30'
    age_group = np.array(["<25" if a < 25 else "25-30" if a < 30 else ">=30" for a in age])
    # merge country and age_group into an array
    sens = np.stack([country, age_group], axis=1).T
    sens_names = ["Country", "Age"]

    # remove "AGE" from features
    features = torch.cat([features[:, :age_idx], features[:, age_idx+1:]], dim=1)
    features = feature_norm(features)

    return g, adj, features, sens, sens_names, masks, labels, feat_names


def preprocess_bail(bail):
    adj, features, idx_train, idx_val, idx_test, labels, sens, feat_names, sens_names \
        = bail.adj(), bail.features(), bail.idx_train(), bail.idx_val(), \
        bail.idx_test(), bail.labels(), bail.sens(), bail.feat_names(), bail.sens_names()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src, dst = adj.coalesce().indices()

    # Create the heterograph
    g = dgl.heterograph({('node', 'edge', 'node'): (src.cpu().numpy(), dst.cpu().numpy())})
    g = g.int().to(device)

    # convert idx_train, idx_val, idx_test to boolean masks
    train_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
    val_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    masks = train_mask, val_mask, test_mask

    # normalize features
    features = feature_norm(features)

    features = features.to(device)
    labels = labels.to(device)
    # if sens is a tensor, convert to numpy array
    if isinstance(sens, torch.Tensor):
        sens = sens.cpu().numpy()

    # race = np.array(["White" if s == 1 else "Other" for s in sens])
    # sens = np.stack([race], axis=1).T
    # sens_names = ["Race"]
    sens_names = ["Race", "Gender"]

    race = sens[0]
    race = np.where(race == 1, 'White', 'Other')
    gender = sens[1]
    # 1 to Male, 0 to Female
    gender = np.where(gender == 0, 'Female', 'Male')
    sens = np.stack([race, gender], axis=1).T

    return g, adj, features, sens, sens_names, masks, labels, feat_names


def preprocess_pokec_n(pokec_n):
    adj, features, idx_train, idx_val, idx_test, labels, sens, feat_names, sens_names \
        = pokec_n.adj(), pokec_n.features(), pokec_n.idx_train(), pokec_n.idx_val(), \
          pokec_n.idx_test(), pokec_n.labels(), pokec_n.sens(), pokec_n.feat_names(), pokec_n.sens_names()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src, dst = adj.coalesce().indices()

    # Create the heterograph
    g = dgl.heterograph({('node', 'edge', 'node'): (src.cpu().numpy(), dst.cpu().numpy())})
    g = g.int().to(device)

    # convert idx_train, idx_val, idx_test to boolean masks
    train_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
    val_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    masks = train_mask, val_mask, test_mask

    # normalize features
    features = feature_norm(features)

    features = features.to(device)
    labels = labels.to(device)
    # if sens is a tensor, convert to numpy array
    if isinstance(sens, torch.Tensor):
        sens = sens.cpu().numpy()

    sens_names = ["Region", "Gender"]

    region = sens[0]
    # region consists of 0 and 1 float values, convert to "0" and "1" string values
    region = region.astype(int).astype(str)
    gender = sens[1]
    # 1 to Male, 0 to Female
    gender = np.where(gender == 0, 'Female', 'Male')
    sens = np.stack([region, gender], axis=1).T

    return g, adj, features, sens, sens_names, masks, labels, feat_names


def preprocess_Pokec_z(pokec_z):
    return preprocess_pokec_n(pokec_z)


def preprocess_credit(credit):
    adj, features, idx_train, idx_val, idx_test, labels, sens, feat_names, sens_names \
        = credit.adj(), credit.features(), credit.idx_train(), credit.idx_val(), \
        credit.idx_test(), credit.labels(), credit.sens(), credit.feat_names(), credit.sens_names()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src, dst = adj.coalesce().indices()

    # Create the heterograph
    g = dgl.heterograph({('node', 'edge', 'node'): (src.cpu().numpy(), dst.cpu().numpy())})
    g = g.int().to(device)

    # convert idx_train, idx_val, idx_test to boolean masks
    train_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
    val_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    masks = train_mask, val_mask, test_mask

    # normalize features
    features = feature_norm(features)

    features = features.to(device)
    labels = labels.to(device)
    # if sens is a tensor, convert to numpy array
    if isinstance(sens, torch.Tensor):
        sens = sens.cpu().numpy()

    age = np.array([">25" if s == 1 else "<=25" for s in sens])
    sens = np.stack([age], axis=1).T
    sens_names = ["Age"]

    return g, adj, features, sens, sens_names, masks, labels, feat_names


def preprocess_german(german):
    adj, features, idx_train, idx_val, idx_test, labels, sens, feat_names \
        = german.adj(), german.features(), german.idx_train(), german.idx_val(), \
        german.idx_test(), german.labels(), german.sens(), german.feat_names()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src, dst = adj.coalesce().indices()

    # Create the heterograph
    g = dgl.heterograph({('node', 'edge', 'node'): (src.cpu().numpy(), dst.cpu().numpy())})
    g = g.int().to(device)

    # convert idx_train, idx_val, idx_test to boolean masks
    train_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
    val_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    masks = train_mask, val_mask, test_mask

    # normalize features
    features = feature_norm(features)

    features = features.to(device)
    labels = labels.to(device)
    # if sens is a tensor, convert to numpy array
    if isinstance(sens, torch.Tensor):
        sens = sens.cpu().numpy()

    gender = np.array(["Male" if s == 0 else "Female" for s in sens])
    sens = np.stack([gender], axis=1).T
    sens_names = ["Gender"]

    return g, adj, features, sens, sens_names, masks, labels, feat_names


def preprocess_facebook(facebook):
    adj, features, idx_train, idx_val, idx_test, labels, sens, feat_names \
        = facebook.adj(), facebook.features(), facebook.idx_train(), facebook.idx_val(), \
        facebook.idx_test(), facebook.labels(), facebook.sens(), facebook.feat_names()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src, dst = adj.coalesce().indices()

    # Create the heterograph
    g = dgl.heterograph({('node', 'edge', 'node'): (src.cpu().numpy(), dst.cpu().numpy())})
    g = g.int().to(device)

    # convert idx_train, idx_val, idx_test to boolean masks
    train_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
    val_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    masks = train_mask, val_mask, test_mask

    # normalize features
    features = feature_norm(features)

    features = features.to(device)
    labels = labels.to(device)
    # if sens is a tensor, convert to numpy array
    if isinstance(sens, torch.Tensor):
        sens = sens.cpu().numpy()

    gender = np.array(["Male" if s == 1 else "Female" for s in sens])
    sens = np.stack([gender], axis=1).T
    sens_names = ["Gender"]

    return g, adj, features, sens, sens_names, masks, labels, feat_names


def data_perturbation(feat, adj, records, feat_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_clone = feat.clone().detach().to(device)
    adj_clone = adj.clone().detach().to(device)
    # Get the row indices and column indices separately from the adj tensor
    indices = adj_clone.coalesce().indices()
    row_indices = indices[0]
    col_indices = indices[1]

    for record in records:
        # attr perturbation
        selected_nodes = record['Nodes']
        attr_indices = []
        for attr in record['Attributes']:
            attr_indices.append(feat_names.index(attr))
        feat_mean = feat.mean(dim=tuple(range(feat.dim() - 1)))
        selected_nodes_tensor = torch.tensor(selected_nodes, dtype=torch.long)  
        attr_indices_tensor = torch.tensor(attr_indices, dtype=torch.long) 
        # Convert row and column indices to a meshgrid of indices
        rows, cols = torch.meshgrid(selected_nodes_tensor, attr_indices_tensor, indexing='ij')
        # Use fancy indexing to set the specified elements to 1
        feat_clone[rows, cols] = feat_mean[attr_indices] 

        # structure perturbation
        if record['Edges']:
            selected_nodes = record['Nodes']
            # Converting the rows_to_zero into a tensor
            selected_nodes_tensor = torch.from_numpy(selected_nodes.astype(int)).to(device)

            # Mask to zero out the selected rows
            row_mask = ~torch.isin(row_indices, selected_nodes_tensor)
            col_mask = ~torch.isin(col_indices, selected_nodes_tensor)
            mask = row_mask & col_mask

            # Apply mask
            row_indices = row_indices[mask]
            col_indices = col_indices[mask]
            # new_values = adj_clone.values()[mask]

            # Determine which diagonal indices need to be added
            needed_diagonals = selected_nodes_tensor
            existing_diagonals = (row_indices == col_indices) & torch.isin(row_indices, needed_diagonals)
            missing_diagonals = needed_diagonals[~torch.isin(needed_diagonals, row_indices[existing_diagonals])]

            # Add missing diagonal elements
            if missing_diagonals.numel() > 0:
                row_indices = torch.cat([row_indices, missing_diagonals])
                col_indices = torch.cat([col_indices, missing_diagonals])
                # new_values = torch.cat([new_values, torch.ones_like(missing_diagonals, dtype=torch.float32)])

    g_new = dgl.heterograph({('node', 'edge', 'node'): (row_indices.cpu().numpy(), col_indices.cpu().numpy())}) 
    g_new = g_new.int().to(device)

    return g_new, feat_clone


def finish_experiment(name, records, features, adj, labels, masks, feat_names, model_new,
                      sens, sens_names, get_embeddings):
    with open(name+'_records.pkl', 'wb') as f:
        pickle.dump(records, f)
    # Open the file in binary read mode
    with open(name+'_records.pkl', 'rb') as f:
        records = pickle.load(f)
    g_new, feat_new = data_perturbation(features, adj, records, feat_names)
    train(g_new, feat_new, labels, masks, model_new, save_path=name+'_retrain.pth')
    # load the model
    model_new.load_state_dict(torch.load(name+'_retrain.pth'))
    model_new.eval()
    e_new = eug.EUG(model_new, adj, features, sens, sens_names, masks, labels, get_embeddings, feat_names=feat_names)

    return e_new