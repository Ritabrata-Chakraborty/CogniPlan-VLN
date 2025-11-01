import torch
import torch.nn as nn
import numpy as np
import math

from .parameter import UTILITY_RANGE
from .utils import check_collision


def compute_sector_features(node_coords, observable_frontiers, updating_map_info):
    """
    Compute features for 8 directional sectors (N, NE, E, SE, S, SW, W, NW) around a node.
    
    Args:
        node_coords: (x, y) coordinates of the node
        observable_frontiers: set of (x, y) frontier coordinates visible from this node
        updating_map_info: MapInfo object for collision checking
    
    Returns:
        sector_features: numpy array of shape (8, 5) with features per sector:
            [count, density, avg_distance, connectivity, orientation]
    """
    # Define 8 sectors: angles in radians (center of each 45° wedge)
    sector_angles = np.array([0, 45, 90, 135, 180, 225, 270, 315]) * np.pi / 180
    sector_width = 45 * np.pi / 180  # 45 degrees in radians
    
    # Sector area (1/8 of a circle with radius UTILITY_RANGE)
    sector_area = np.pi * UTILITY_RANGE ** 2 / 8
    
    # Initialize features: 8 sectors × 5 features
    sector_features = np.zeros((8, 5))
    
    if len(observable_frontiers) == 0:
        # No frontiers: set orientation only
        sector_features[:, 4] = sector_angles / (2 * np.pi)  # Normalize to [0, 1]
        return sector_features
    
    # Convert frontiers to numpy array
    frontiers = np.array(list(observable_frontiers)).reshape(-1, 2)
    
    # Compute relative positions
    relative_positions = frontiers - node_coords
    distances = np.linalg.norm(relative_positions, axis=1)
    
    # Compute angles (0 = North = +Y direction, clockwise)
    # We need to adjust: standard atan2 gives 0 at +X, we want 0 at +Y
    angles = np.arctan2(relative_positions[:, 0], relative_positions[:, 1])  # x, y for correct orientation
    angles = np.mod(angles, 2 * np.pi)  # Ensure [0, 2π]
    
    # Assign each frontier to a sector
    for sector_idx, sector_angle in enumerate(sector_angles):
        # Determine angular boundaries of this sector
        angle_min = sector_angle - sector_width / 2
        angle_max = sector_angle + sector_width / 2
        
        # Handle wrap-around at 0/360 degrees
        if angle_min < 0:
            in_sector = (angles >= (angle_min + 2 * np.pi)) | (angles < angle_max)
        elif angle_max > 2 * np.pi:
            in_sector = (angles >= angle_min) | (angles < (angle_max - 2 * np.pi))
        else:
            in_sector = (angles >= angle_min) & (angles < angle_max)
        
        sector_frontiers = frontiers[in_sector]
        sector_distances = distances[in_sector]
        
        # Feature 1: Count
        count = len(sector_frontiers)
        sector_features[sector_idx, 0] = count
        
        # Feature 2: Density (count per unit area)
        density = count / sector_area if sector_area > 0 else 0
        sector_features[sector_idx, 1] = density
        
        # Feature 3: Average distance
        if count > 0:
            avg_distance = np.mean(sector_distances)
            sector_features[sector_idx, 2] = avg_distance / UTILITY_RANGE  # Normalize
        else:
            sector_features[sector_idx, 2] = 0  # No frontiers
        
        # Feature 4: Connectivity (ratio of frontiers with line-of-sight)
        if count > 0:
            connected_count = 0
            for frontier in sector_frontiers:
                collision = check_collision(node_coords, frontier, updating_map_info)
                if not collision:
                    connected_count += 1
            connectivity = connected_count / count
            sector_features[sector_idx, 3] = connectivity
        else:
            sector_features[sector_idx, 3] = 0
        
        # Feature 5: Sector orientation (normalized angle)
        sector_features[sector_idx, 4] = sector_angle / (2 * np.pi)
    
    return sector_features


class SingleHeadCrossAttention(nn.Module):
    """
    Single-head cross-attention mechanism for sector features.
    Query: node positions (2D)
    Keys/Values: sector features (8 sectors × 5 features)
    """
    def __init__(self, query_dim, key_dim, embed_dim):
        super(SingleHeadCrossAttention, self).__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.norm_factor = 1 / math.sqrt(embed_dim)
        
        # Projection matrices
        self.w_query = nn.Linear(query_dim, embed_dim)
        self.w_key = nn.Linear(key_dim, embed_dim)
        self.w_value = nn.Linear(key_dim, embed_dim)
        
    def forward(self, query, key_value):
        """
        Args:
            query: (batch, n_nodes, query_dim) - node positions
            key_value: (batch, n_nodes, n_sectors, key_dim) - sector features
        
        Returns:
            output: (batch, n_nodes, embed_dim) - attention-weighted sector information
        """
        batch_size, n_nodes, _ = query.size()
        _, _, n_sectors, _ = key_value.size()
        
        # Project query: (batch, n_nodes, embed_dim)
        Q = self.w_query(query)
        
        # Reshape key_value for projection: (batch, n_nodes, n_sectors, key_dim) 
        # -> (batch * n_nodes, n_sectors, key_dim)
        kv_reshaped = key_value.reshape(batch_size * n_nodes, n_sectors, self.key_dim)
        
        # Project keys and values
        K = self.w_key(kv_reshaped)  # (batch * n_nodes, n_sectors, embed_dim)
        V = self.w_value(kv_reshaped)  # (batch * n_nodes, n_sectors, embed_dim)
        
        # Reshape back: (batch, n_nodes, n_sectors, embed_dim)
        K = K.reshape(batch_size, n_nodes, n_sectors, self.embed_dim)
        V = V.reshape(batch_size, n_nodes, n_sectors, self.embed_dim)
        
        # Compute attention scores: Q @ K^T
        # Q: (batch, n_nodes, embed_dim) -> (batch, n_nodes, 1, embed_dim)
        # K: (batch, n_nodes, n_sectors, embed_dim) -> (batch, n_nodes, embed_dim, n_sectors)
        Q_expanded = Q.unsqueeze(2)  # (batch, n_nodes, 1, embed_dim)
        K_transposed = K.transpose(2, 3)  # (batch, n_nodes, embed_dim, n_sectors)
        
        # Attention scores: (batch, n_nodes, 1, n_sectors)
        attn_scores = torch.matmul(Q_expanded, K_transposed) * self.norm_factor
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention to values: (batch, n_nodes, 1, n_sectors) @ (batch, n_nodes, n_sectors, embed_dim)
        # -> (batch, n_nodes, 1, embed_dim)
        output = torch.matmul(attn_weights, V)
        output = output.squeeze(2)  # (batch, n_nodes, embed_dim)
        
        return output


class SectorFrontierEncoder(nn.Module):
    """
    Encodes sector-wise frontier information using cross-attention.
    Outputs a fixed-size representation enriched with directional frontier features.
    """
    def __init__(self, position_dim=2, sector_feature_dim=5, n_sectors=8, output_dim=26):
        super(SectorFrontierEncoder, self).__init__()
        self.position_dim = position_dim
        self.sector_feature_dim = sector_feature_dim
        self.n_sectors = n_sectors
        self.output_dim = output_dim
        
        # Cross-attention module
        embed_dim = 32  # Internal embedding dimension for attention
        self.cross_attention = SingleHeadCrossAttention(
            query_dim=position_dim,
            key_dim=sector_feature_dim,
            embed_dim=embed_dim
        )
        
        # Project attention output to desired output dimension
        self.output_projection = nn.Linear(embed_dim, output_dim)
        
    def forward(self, node_positions, sector_features):
        """
        Args:
            node_positions: (batch, n_nodes, 2) - node coordinates
            sector_features: (batch, n_nodes, 8, 5) - sector features per node
        
        Returns:
            output: (batch, n_nodes, output_dim) - enriched node features
        """
        # Apply cross-attention
        attended_features = self.cross_attention(node_positions, sector_features)
        
        # Project to output dimension
        output = self.output_projection(attended_features)
        
        return output

