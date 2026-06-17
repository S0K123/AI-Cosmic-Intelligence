import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

from data_loader import load_data
from feature_engineering import clean_and_prepare_data
from train_model import create_habitability_label

class StarSystemDataset(Dataset):
    def __init__(self, root, filename, system_graphs):
        self.system_graphs = system_graphs
        super(StarSystemDataset, self).__init__(root, filename)

    def len(self):
        return len(self.system_graphs)

    def get(self, idx):
        return self.system_graphs[idx]

def create_graph_dataset(df):
    """Constructs a list of graph data objects, one for each star system."""
    system_graphs = []
    # Group by star system (hostname)
    for system_name, system_df in df.groupby('hostname'):
        if system_df.empty:
            continue

        num_planets = len(system_df)
        # Node features: star (1) + planets (num_planets)
        # Star features: temp, radius, luminosity
        # Planet features: radius, period, semi-major axis
        star_features = system_df[['star_temp_k', 'star_radius_solar', 'star_luminosity']].iloc[0].values
        planet_features = system_df[['planet_radius', 'orbital_period', 'orbit_semi_major_axis']].values
        
        # Pad features to be the same size (3)
        star_node_features = np.pad(star_features, (0, 0), 'constant')
        planet_node_features = planet_features

        node_features = torch.tensor(np.vstack([star_node_features, planet_node_features]), dtype=torch.float)

        # Edges: star-planet and planet-planet
        edge_list = []
        # Star to each planet
        for i in range(num_planets):
            edge_list.append([0, i + 1])
            edge_list.append([i + 1, 0])
        # Planet to every other planet
        for i in range(num_planets):
            for j in range(i + 1, num_planets):
                edge_list.append([i + 1, j + 1])
                edge_list.append([j + 1, i + 1])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # Labels (for planet nodes)
        labels = torch.tensor(system_df['habitability_label'].values, dtype=torch.float)
        # Star has no label, so we use a placeholder (-1)
        y = torch.cat([torch.tensor([-1], dtype=torch.float), labels])

        system_graphs.append(Data(x=node_features, edge_index=edge_index, y=y))
    
    return system_graphs

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.classifier = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # We apply the classifier to each node
        x = self.classifier(x)
        return x

def get_trained_gnn(df):
    """Trains and returns a GNN model."""
    system_graphs = create_graph_dataset(df)
    
    train_graphs, test_graphs = train_test_split(system_graphs, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    
    model = GCN(num_node_features=3, num_classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    print("\n--- GNN Model Training ---")
    for epoch in range(50): # Reduced epochs for faster integration
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            planet_mask = batch.y != -1
            loss = criterion(out[planet_mask], batch.y[planet_mask].unsqueeze(1))
            loss.backward()
            optimizer.step()
    
    print("--- GNN Model Trained ---")
    return model

if __name__ == "__main__":
    raw_df = load_data()
    if raw_df is not None:
        if 'default_flag' in raw_df.columns:
            raw_df = raw_df[raw_df['default_flag'] == 1].copy()
    
        prepared_df = clean_and_prepare_data(raw_df)
        labeled_df = create_habitability_label(prepared_df)
        
        # This part is now for standalone testing of the GNN
        train_and_evaluate_gnn(labeled_df)
