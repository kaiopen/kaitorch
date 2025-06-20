import torch

from kaitorch.data import squared_euclidean_distance
from kaitorch.zoo.pointnet2 import PointNetSampler, PointNetNeighborOperator


sa = PointNetSampler(
    batch_size=1,
    num_center=3,
    radius=5,
    num_neighbor=5,
    operator=PointNetNeighborOperator(
        in_channels=4,
        mlp=[64, 128, 256]
    ),
    out_channels=128
).to(torch.device('cuda:0'))
positions = torch.tensor(
    [[
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5],
        [6, 6, 6],
        [7, 7, 7],
        [8, 8, 8],
        [9, 9, 9],
        [1, 5, 5],
        [5, 6, 2],
        [8, 9, 0],
        [1, 2, 5],
        [3, 3, 7],
        [5, 5, 5],
        [9, 8, 8],
        [2, 2, 7],
        [9, 2, 6],
        [0, 9, 1],
        [7, 2, 5.]
    ]]
).float().to(torch.device('cuda:0'))
features = torch.arange(
    21, dtype=float
).reshape(1, 21, 1).float().to(torch.device('cuda:0'))
centers = torch.tensor(
    [[
        [0, 0, 0],
        [3, 3, 3],
        [7, 7, 7.]
    ]]
).float().to(torch.device('cuda:0'))
distances = squared_euclidean_distance(
    centers.unsqueeze(2), positions.unsqueeze(1)
)

print('------')
print('positions:', positions.shape)
print('features:', features.shape)
print('distances:', distances.shape)
print('------')

new_features = sa(positions, features, centers, distances)
print('new features:', new_features.shape)
