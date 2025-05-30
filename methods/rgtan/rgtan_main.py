import os
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.dataloading import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from scipy.io import loadmat
import scipy.sparse as sp
from tqdm import tqdm
from . import *
from .rgtan_lpa import load_lpa_subtensor
from .rgtan_model import RGTAN


def rgtan_main(feat_df, graph, train_idx, test_idx, labels, args, cat_features, neigh_features: pd.DataFrame, nei_att_head):
    """
    Main training and evaluation function for RGTAN model.
    
    This function implements the complete training pipeline including:
    - K-fold cross-validation
    - Mini-batch training with neighbor sampling
    - Early stopping
    - Model evaluation on test set
    
    Args:
        feat_df: DataFrame containing node features
        graph: DGL graph object
        train_idx: Indices of training nodes
        test_idx: Indices of test nodes
        labels: Node labels
        args: Dictionary of configuration parameters
        cat_features: List of categorical feature column names
        neigh_features: DataFrame containing neighborhood risk statistics
        nei_att_head: Number of attention heads for neighborhood features
        
    Returns:
        None (prints evaluation metrics)
    """
    # torch.autograd.set_detect_anomaly(True)
    device = args['device']
    graph = graph.to(device)
    oof_predictions = torch.from_numpy(
        np.zeros([len(feat_df), 2])).float().to(device)
    test_predictions = torch.from_numpy(
        np.zeros([len(feat_df), 2])).float().to(device)
    kfold = StratifiedKFold(
        n_splits=args['n_fold'], shuffle=True, random_state=args['seed'])

    y_target = labels.iloc[train_idx].values
    num_feat = torch.from_numpy(feat_df.values).float().to(device)
    
    # Handle categorical features - for creditcard dataset they might not be needed
    # since embeddings are already in the model
    if cat_features and len(cat_features) > 0:
        cat_feat = {}
        for col in cat_features:
            if col in feat_df.columns:
                # Original categorical column exists
                cat_feat[col] = torch.from_numpy(feat_df[col].values).long().to(device)
            elif col + '_encoded' in feat_df.columns:
                # Use the encoded version
                cat_feat[col] = torch.from_numpy(feat_df[col + '_encoded'].values).long().to(device)
            else:
                # For creditcard dataset, categorical features might be preprocessed
                # Create dummy indices for now
                print(f"Info: Categorical feature {col} not in feat_df, using indices")
                cat_feat[col] = torch.arange(len(feat_df)).long().to(device)
    else:
        # No categorical features
        cat_feat = {}

    neigh_padding_dict = {}
    nei_feat = []
    if isinstance(neigh_features, pd.DataFrame):  # otherwise []
        # if null it is []
        nei_feat = {col: torch.from_numpy(neigh_features[col].values).to(torch.float32).to(
            device) for col in neigh_features.columns}
        
    y = labels
    labels = torch.from_numpy(y.values).long().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    for fold, (trn_idx, val_idx) in enumerate(kfold.split(feat_df.iloc[train_idx], y_target)):
        print(f'Training fold {fold + 1}')
        trn_ind, val_ind = torch.from_numpy(np.array(train_idx)[trn_idx]).long().to(
            device), torch.from_numpy(np.array(train_idx)[val_idx]).long().to(device)

        train_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        train_dataloader = DataLoader(graph,
                                          trn_ind,
                                          train_sampler,
                                          device=device,
                                          use_ddp=False,
                                          batch_size=args['batch_size'],
                                          shuffle=True,
                                          drop_last=False,
                                          num_workers=0
                                          )
        val_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        val_dataloader = DataLoader(graph,
                                        val_ind,
                                        val_sampler,
                                        use_ddp=False,
                                        device=device,
                                        batch_size=args['batch_size'],
                                        shuffle=True,
                                        drop_last=False,
                                        num_workers=0,
                                        )
        model = RGTAN(in_feats=feat_df.shape[1],
                      hidden_dim=args['hid_dim']//4,
                      n_classes=2,
                      heads=args.get('conv_heads', [4]*args['n_layers']),
                      activation=nn.PReLU(),
                      n_layers=args['n_layers'],
                      drop=args['dropout'],
                      device=device,
                      gated=args['gated'],
                      ref_df=feat_df,
                      cat_features=cat_feat,
                      neigh_features=nei_feat,
                      nei_att_head=nei_att_head).to(device)
        lr = args['lr'] * np.sqrt(args['batch_size']/1024)
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=args['wd'])
        lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=args.get('lr_scheduler', {}).get('milestones', [4000, 12000]), 
                                   gamma=args.get('lr_scheduler', {}).get('gamma', 0.3))

        earlystoper = early_stopper(
            patience=args['early_stopping'], verbose=True)
        start_epoch, max_epochs = 0, args.get('max_training_epochs', 2000)
        for epoch in range(start_epoch, args['max_epochs']):
            train_loss_list = []
            # train_acc_list = []
            model.train()
            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                # print(f"loading batch data...")
                batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                                                                                                                       seeds, input_nodes, device, blocks)
                # print(f"load {step}")

                # batch_neighstat_inputs: {"degree":(|batch|, degree_dim)}

                blocks = [block.to(device) for block in blocks]
                train_batch_logits = model(
                    blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)
                mask = batch_labels == 2
                train_batch_logits = train_batch_logits[~mask]
                batch_labels = batch_labels[~mask]
                # batch_labels[mask] = 0

                train_loss = loss_fn(train_batch_logits, batch_labels)
                # backward
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                train_loss_list.append(train_loss.cpu().detach().numpy())

                if step % 10 == 0:
                    tr_batch_pred = torch.sum(torch.argmax(train_batch_logits.clone(
                    ).detach(), dim=1) == batch_labels) / batch_labels.shape[0]
                    score = torch.softmax(train_batch_logits.clone().detach(), dim=1)[
                        :, 1].cpu().numpy()
                    try:
                        print('In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, '
                              'train_ap:{:.4f}, train_acc:{:.4f}, train_auc:{:.4f}'.format(epoch, step,
                                                                                           np.mean(
                                                                                               train_loss_list),
                                                                                           average_precision_score(
                                                                                               batch_labels.cpu().numpy(), score),
                                                                                           tr_batch_pred.detach(),
                                                                                           roc_auc_score(batch_labels.cpu().numpy(), score)))
                    except ValueError as e:
                        # This can happen when all labels are the same in a batch
                        print(f'Metrics calculation failed at epoch {epoch}, batch {step}: {e}')

            # mini-batch for validation
            val_loss_list = 0
            val_acc_list = 0
            val_all_list = 0
            model.eval()
            with torch.no_grad():
                for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                    batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                                                                                                                           seeds, input_nodes, device, blocks)

                    blocks = [block.to(device) for block in blocks]
                    val_batch_logits = model(
                        blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)
                    oof_predictions[seeds] = val_batch_logits
                    mask = batch_labels == 2
                    val_batch_logits = val_batch_logits[~mask]
                    batch_labels = batch_labels[~mask]
                    # batch_labels[mask] = 0
                    val_loss_list = val_loss_list + \
                        loss_fn(val_batch_logits, batch_labels)
                    # val_all_list += 1
                    val_batch_pred = torch.sum(torch.argmax(
                        val_batch_logits, dim=1) == batch_labels) / torch.tensor(batch_labels.shape[0])
                    val_acc_list = val_acc_list + val_batch_pred * \
                        torch.tensor(
                            batch_labels.shape[0])  # how many in this batch is right!
                    val_all_list = val_all_list + \
                        batch_labels.shape[0]  # how many val nodes
                    if step % 10 == 0:
                        score = torch.softmax(val_batch_logits.clone().detach(), dim=1)[
                            :, 1].cpu().numpy()
                        try:
                            print('In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, val_ap:{:.4f}, '
                                  'val_acc:{:.4f}, val_auc:{:.4f}'.format(epoch,
                                                                          step,
                                                                          val_loss_list/val_all_list,
                                                                          average_precision_score(
                                                                              batch_labels.cpu().numpy(), score),
                                                                          val_batch_pred.detach(),
                                                                          roc_auc_score(batch_labels.cpu().numpy(), score)))
                        except ValueError as e:
                            # This can happen when all labels are the same in a batch
                            print(f'Metrics calculation failed at epoch {epoch}, batch {step}: {e}')

            # val_acc_list/val_all_list, model)
            earlystoper.earlystop(val_loss_list/val_all_list, model)
            if earlystoper.is_earlystop:
                print("Early Stopping!")
                break
        print("Best val_loss is: {:.7f}".format(earlystoper.best_cv))
        test_ind = torch.from_numpy(np.array(test_idx)).long().to(device)
        test_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        test_dataloader = DataLoader(graph,
                                         test_ind,
                                         test_sampler,
                                         use_ddp=False,
                                         device=device,
                                         batch_size=args['batch_size'],
                                         shuffle=True,
                                         drop_last=False,
                                         num_workers=0,
                                         )
        b_model = earlystoper.best_model.to(device)
        b_model.eval()
        with torch.no_grad():
            for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
                # print(input_nodes)
                batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                                                                                                                       seeds, input_nodes, device, blocks)

                blocks = [block.to(device) for block in blocks]
                test_batch_logits = b_model(
                    blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)
                test_predictions[seeds] = test_batch_logits
                test_batch_pred = torch.sum(torch.argmax(
                    test_batch_logits, dim=1) == batch_labels) / torch.tensor(batch_labels.shape[0])
                if step % 10 == 0:
                    print('In test batch:{:04d}'.format(step))
    mask = y_target == 2
    y_target[mask] = 0
    my_ap = average_precision_score(y_target, torch.softmax(
        oof_predictions, dim=1).cpu()[train_idx, 1])
    print("NN out of fold AP is:", my_ap)
    b_models, val_gnn_0, test_gnn_0 = earlystoper.best_model.to(
        'cpu'), oof_predictions, test_predictions

    test_score = torch.softmax(test_gnn_0, dim=1)[test_idx, 1].cpu().numpy()
    y_target = labels[test_idx].cpu().numpy()
    test_score1 = torch.argmax(test_gnn_0, dim=1)[test_idx].cpu().numpy()

    mask = y_target != 2
    test_score = test_score[mask]
    y_target = y_target[mask]
    test_score1 = test_score1[mask]

    print("test AUC:", roc_auc_score(y_target, test_score))
    print("test f1:", f1_score(y_target, test_score1, average="macro"))
    print("test AP:", average_precision_score(y_target, test_score))


def load_rgtan_data(dataset: str, test_size: float):
    """
    Load and preprocess data for RGTAN model.
    
    This function handles data loading for different datasets:
    - S-FFSD: Financial fraud dataset with transaction features
    - Yelp: Review fraud dataset
    - Amazon: Product review fraud dataset
    
    The function constructs a graph based on shared attributes (for S-FFSD)
    or uses pre-computed adjacency lists (for Yelp/Amazon).
    
    Args:
        dataset: Name of the dataset ('S-FFSD', 'yelp', or 'amazon')
        test_size: Proportion of data to use for testing
        
    Returns:
        tuple: (feat_data, labels, train_idx, test_idx, graph, cat_features, neigh_features)
            - feat_data: DataFrame of node features
            - labels: Series of node labels
            - train_idx: List of training node indices
            - test_idx: List of test node indices
            - graph: DGL graph object
            - cat_features: List of categorical feature names
            - neigh_features: DataFrame of neighborhood statistics (if available)
    """
    # prefix = "./antifraud/data/"
    prefix = "data/"
    if dataset == 'S-FFSD':
        cat_features = ["Target", "Location", "Type"]

        
        df = pd.read_csv(prefix + "S-FFSDneofull.csv")
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        #####
        neigh_features = []
        #####
        data = df[df["Labels"] <= 2]
        data = data.reset_index(drop=True)
        out = []
        alls = []
        allt = []
        pair = ["Source", "Target", "Location", "Type"]
        for column in pair:
            src, tgt = [], []
            edge_per_trans = 3
            for c_id, c_df in tqdm(data.groupby(column), desc=column):
                c_df = c_df.sort_values(by="Time")
                df_len = len(c_df)
                sorted_idxs = c_df.index
                src.extend([sorted_idxs[i] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
                tgt.extend([sorted_idxs[i+j] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
            alls.extend(src)
            allt.extend(tgt)
        alls = np.array(alls)
        allt = np.array(allt)
        g = dgl.graph((alls, allt))
        
        # Add self-loops to handle isolated nodes
        g = dgl.add_self_loop(g)
        
        cal_list = ["Source", "Target", "Location", "Type"]
        for col in cal_list:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].apply(str).values)
        feat_data = data.drop("Labels", axis=1)
        labels = data["Labels"]

        #######
        g.ndata['label'] = torch.from_numpy(
            labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        #######

        graph_path = prefix+"graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])
        index = list(range(len(labels)))

        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=0.6,
                                                                random_state=2, shuffle=True)
        feat_neigh = pd.read_csv(
            prefix + "S-FFSD_neigh_feat.csv")
        print("neighborhood feature loaded for nn input.")
        neigh_features = feat_neigh

    elif dataset == 'yelp':
        cat_features = []
        neigh_features = []
        data_file = loadmat(prefix + 'YelpChi.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        # load the preprocessed adj_lists
        with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        index = list(range(len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=test_size,
                                                                random_state=2, shuffle=True)
        src = []
        tgt = []
        for i in homo:
            for j in homo[i]:
                src.append(i)
                tgt.append(j)
        src = np.array(src)
        tgt = np.array(tgt)
        g = dgl.graph((src, tgt))
        # Add self-loops to handle isolated nodes
        g = dgl.add_self_loop(g)
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])

        try:
            feat_neigh = pd.read_csv(
                prefix + "yelp_neigh_feat.csv")
            print("neighborhood feature loaded for nn input.")
            neigh_features = feat_neigh
        except FileNotFoundError:
            print("no neighborhood feature used - file not found.")
        except Exception as e:
            print(f"Error loading neighborhood features: {e}")

    elif dataset == 'amazon':
        cat_features = []
        neigh_features = []
        data_file = loadmat(prefix + 'Amazon.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        # load the preprocessed adj_lists
        with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        index = list(range(3305, len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:],
                                                                test_size=test_size, random_state=2, shuffle=True)
        src = []
        tgt = []
        for i in homo:
            for j in homo[i]:
                src.append(i)
                tgt.append(j)
        src = np.array(src)
        tgt = np.array(tgt)
        g = dgl.graph((src, tgt))
        # Add self-loops to handle isolated nodes
        g = dgl.add_self_loop(g)
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])
        try:
            feat_neigh = pd.read_csv(
                prefix + "amazon_neigh_feat.csv")
            print("neighborhood feature loaded for nn input.")
            neigh_features = feat_neigh
        except FileNotFoundError:
            print("no neighborhood feature used - file not found.")
        except Exception as e:
            print(f"Error loading neighborhood features: {e}")

    elif dataset == 'creditcard':
        import itertools
        from collections import defaultdict
        
        cat_features = []
        neigh_features = []
        
        # First check if MAT format exists (preferred format)
        mat_file = prefix + 'CreditCard.mat'
        if os.path.exists(mat_file):
            print(f"Loading CreditCard dataset from {mat_file}...")
            data_file = loadmat(mat_file)
            labels = pd.DataFrame(data_file['label'].flatten())[0]
            feat_data = pd.DataFrame(data_file['features'].todense().A)
            
            # Build adjacency list from homo matrix
            homo_adj = data_file['homo']
            homo = defaultdict(list)
            rows, cols = sp.find(homo_adj)[:2]
            for i, j in zip(rows, cols):
                if i != j:  # Skip self-loops for adjacency list
                    homo[i].append(j)
            
            # Save adjacency list for future use
            adj_file = prefix + 'creditcard_homo_adjlists.pickle'
            with open(adj_file, 'wb') as f:
                pickle.dump(dict(homo), f)
            
            index = list(range(len(labels)))
            train_idx, test_idx, y_train, y_test = train_test_split(
                index, labels, stratify=labels, test_size=test_size, 
                random_state=2, shuffle=True
            )
            
            # Build DGL graph
            src = []
            tgt = []
            for i in homo:
                for j in homo[i]:
                    src.append(i)
                    tgt.append(j)
            src = np.array(src)
            tgt = np.array(tgt)
            g = dgl.graph((src, tgt))
            g = dgl.add_self_loop(g)
            g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
            g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)
            
            graph_path = prefix + "graph-{}.bin".format(dataset)
            dgl.data.utils.save_graphs(graph_path, [g])
            
            print(f"Loaded CreditCard dataset: {len(labels)} nodes, {len(src)} edges")
            print(f"Feature dimension: {feat_data.shape[1]}")
            print(f"Fraud rate: {labels.mean():.2%}")
            
        else:
            # Fall back to CSV preprocessing approach
            print(f"CreditCard.mat not found. Checking for preprocessed CSV data...")
            
            # Check if preprocessed data exists (try realistic version first)
            preprocessed_file = prefix + 'creditcard_realistic_preprocessed.csv'
            adj_file = prefix + 'creditcard_realistic_adjlists.pickle'
            neigh_file = prefix + 'creditcard_realistic_neigh_feat.csv'
            split_file = prefix + 'creditcard_realistic_splits.pickle'
            
            # Fall back to original preprocessing if realistic doesn't exist
            if not os.path.exists(preprocessed_file):
                preprocessed_file = prefix + 'creditcard_preprocessed.csv'
                adj_file = prefix + 'creditcard_homo_adjlists.pickle'
                neigh_file = prefix + 'creditcard_neigh_feat.csv'
                split_file = None
            
            if os.path.exists(preprocessed_file) and os.path.exists(adj_file):
                print("Loading preprocessed creditcard data...")
                df = pd.read_csv(preprocessed_file)
                
                # Load adjacency lists
                with open(adj_file, 'rb') as f:
                    adjacency_lists = pickle.load(f)
                
                # Try to load neighborhood features
                if os.path.exists(neigh_file):
                    neigh_features = pd.read_csv(neigh_file)
                    print("Loaded neighborhood features.")
                    
                # When using preprocessed data, categorical features are already encoded
                # For RGTAN with preprocessed data, we don't need separate categorical features
                # since they're already included in the feature matrix as encoded values
                cat_features = []
                encoded_features = [col for col in df.columns if col.endswith('_encoded')]
                print(f"Found {len(encoded_features)} encoded categorical features from preprocessing")
            else:
                print("Preprocessed data not found. Loading raw data...")
                print("Consider running: python feature_engineering/preprocess_creditcard.py")
                # Read the raw creditcard dataset
                df = pd.read_csv(prefix + 'vod_creditcard.csv')
        
            # Convert labels if not already done
            if 'Labels' not in df.columns:
                if 'auth_msg' in df.columns:
                    # Create labels based on auth_msg
                    print("Creating labels from auth_msg field...")
                    
                    # Define fraud patterns
                    fraud_patterns = [
                        'DECLINE', 'INSUFF FUNDS', 'CALL', 'INVALID MERCHANT',
                        'BLOCKED', 'DECLINE SH', 'DECLINE SC', 'TERM ID ERROR',
                        'INVALID TRANS', 'ACCT LENGTH ERR', 'STOLEN CARD',
                        'LOST CARD', 'PICKUP CARD', 'FRAUD', 'SECURITY VIOLATION',
                        'RESTRICTED CARD', 'EXPIRED CARD'
                    ]
                    
                    # Default to non-fraud
                    df['Labels'] = 0
                    
                    # Mark as fraud if auth_msg contains any fraud pattern
                    for pattern in fraud_patterns:
                        mask = df['auth_msg'].str.contains(pattern, case=False, na=False)
                        df.loc[mask, 'Labels'] = 1
                    
                    # Ensure approved transactions are marked as non-fraud
                    approved_mask = df['auth_msg'].str.contains('APPROVED', case=False, na=False)
                    df.loc[approved_mask, 'Labels'] = 0
                    
                    print(f"Label distribution - Fraud: {df['Labels'].mean():.2%}")
                else:
                    print("Warning: No auth_msg column found, using IS_TARGETED if available")
                    if 'IS_TARGETED' in df.columns:
                        df['Labels'] = df['IS_TARGETED'].map({'yes': 1, 'no': 0})
                    else:
                        raise ValueError("Neither auth_msg nor IS_TARGETED columns found for labels")
            
            # Handle datetime columns
            date_columns = ['issue_date', 'capture_date', 'created_date', 'updated_date']
            for col in date_columns:
                if col in df.columns:
                    # Convert to datetime, handling NULL values
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    # Convert to seconds since first transaction
                    if not df[col].isna().all():
                        min_date = df[col].min()
                        df[col + '_seconds'] = (df[col] - min_date).dt.total_seconds()
                        df[col + '_seconds'] = df[col + '_seconds'].fillna(0)
            
            # Select numerical features
            num_features = ['amount', 'issue_date_seconds']
            
            # Select categorical features for embeddings - only use columns that exist
            potential_cat_features = ['trans_status_msg_id', 'site_tag_id', 'origin_id', 
                                     'currency_id', 'card_type_id', 'processor_id', 
                                     'trans_status_code', 'BRAND', 'DEBITCREDIT', 'CARDTYPE']
            
            # Filter to only columns that exist in the dataframe
            cat_features_list = [col for col in potential_cat_features if col in df.columns]
            
            print(f"Available categorical features: {cat_features_list}")
            
            # Handle missing values in categorical features
            for col in cat_features_list:
                df[col] = df[col].fillna(-1)
                # Convert to string first to handle mixed types
                df[col] = df[col].astype(str)
            
            # Build graph - use preprocessed adjacency lists if available
            if 'adjacency_lists' in locals():
                # Use preprocessed adjacency lists
                g = dgl.graph([])
                g.add_nodes(len(df))
                edge_src = []
                edge_dst = []
                for src, dsts in adjacency_lists.items():
                    for dst in dsts:
                        edge_src.append(src)
                        edge_dst.append(dst)
                if edge_src:
                    g.add_edges(edge_src, edge_dst)
                # Add self-loops to handle isolated nodes
                g = dgl.add_self_loop(g)
            else:
                # Build graph from scratch
                g = dgl.graph([])
                g.add_nodes(len(df))
                
                # Define key columns for edge creation
                key_columns = ['card_number', 'member_id', 'customer_ip', 'customer_email', 'BIN']
                
                edge_src = []
                edge_dst = []
                
                for col in key_columns:
                    if col in df.columns:
                        # Group by the key column
                        mapping = defaultdict(list)
                        for idx, val in enumerate(df[col]):
                            if pd.notna(val) and val != '':  # Skip null/empty values
                                mapping[val].append(idx)
                        
                        # Create edges between transactions with same key value
                        for idxs in mapping.values():
                            if len(idxs) > 1:
                                # Create all permutations of edges
                                for i, j in itertools.permutations(idxs, 2):
                                    edge_src.append(i)
                                    edge_dst.append(j)
                
                # Add edges to graph
                if edge_src:
                    g.add_edges(edge_src, edge_dst)
                
                # Add self-loops to handle isolated nodes
                g = dgl.add_self_loop(g)
            
            print(f"Created graph with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")
            
            # Prepare feature data
            # Use scaled features if available (from preprocessing)
            scaled_cols = [col for col in df.columns if col.endswith('_scaled')]
            if scaled_cols:
                print(f"Using {len(scaled_cols)} scaled features from preprocessing")
                feat_cols = scaled_cols
                
                # Also include encoded categorical features if they exist
                encoded_cols = [col for col in df.columns if col.endswith('_encoded')]
                if encoded_cols:
                    print(f"Using {len(encoded_cols)} encoded categorical features from preprocessing")
                    feat_cols.extend(encoded_cols)
            else:
                # Build features from scratch
                # Combine numerical features
                feat_cols = []
                for col in num_features:
                    if col in df.columns:
                        feat_cols.append(col)
                
                # Add encoded categorical features
                for col in cat_features_list:
                    if col + '_encoded' not in df.columns:
                        le = LabelEncoder()
                        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                    feat_cols.append(col + '_encoded')
            
            # Create feat_data with all necessary columns for RGTAN
            # Include both the feature columns AND the original categorical columns
            all_cols = feat_cols + [col for col in cat_features if col in df.columns and col not in feat_cols]
            feat_data = df[all_cols].copy()
            
            # Ensure categorical columns are properly typed
            for col in cat_features:
                if col in feat_data.columns:
                    # Convert to numeric, filling non-numeric values with -1
                    feat_data[col] = pd.to_numeric(feat_data[col], errors='coerce').fillna(-1).astype(int)
            
            # Fill any remaining NaN values
            feat_data = feat_data.fillna(0)
            labels = df['Labels']
            
            # Add node features to graph (only the numerical features)
            g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
            g.ndata['feat'] = torch.from_numpy(df[feat_cols].to_numpy()).to(torch.float32)
            
            # Add self-loops to handle isolated nodes
            g = dgl.add_self_loop(g)
            
            # Save graph
            graph_path = prefix + "graph-{}.bin".format(dataset)
            dgl.data.utils.save_graphs(graph_path, [g])
            
            # Split data - use saved splits if available (for realistic preprocessing)
            if split_file and os.path.exists(split_file):
                print("Using saved train/test splits...")
                with open(split_file, 'rb') as f:
                    splits = pickle.load(f)
                    train_mask = splits['train_mask']
                    train_idx = df[train_mask].index.tolist()
                    test_idx = df[~train_mask].index.tolist()
            else:
                # Fall back to time-based split
                print("Creating time-based split...")
                df_sorted = df.sort_values('issue_date')
                split_idx = int(len(df_sorted) * 0.7)
                
                train_idx = df_sorted.index[:split_idx].tolist()
                test_idx = df_sorted.index[split_idx:].tolist()
            
            print(f"Train set: {len(train_idx)} transactions, Test set: {len(test_idx)} transactions")
            print(f"Train fraud rate: {labels.iloc[train_idx].mean():.2%}")
            print(f"Test fraud rate: {labels.iloc[test_idx].mean():.2%}")
            
            # Try to load neighborhood features if available
            try:
                feat_neigh = pd.read_csv(prefix + "creditcard_neigh_feat.csv")
                print("neighborhood feature loaded for nn input.")
                neigh_features = feat_neigh
            except FileNotFoundError:
                print("no neighborhood feature used - file not found.")
                neigh_features = []
            except Exception as e:
                print(f"Error loading neighborhood features: {e}")
                neigh_features = []
            
            # For creditcard with raw data, set categorical features
            if 'cat_features' not in locals():
                cat_features = [col for col in cat_features_list if col in df.columns]

    return feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features
