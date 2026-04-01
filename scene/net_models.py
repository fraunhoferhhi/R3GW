import torch
from torch import nn
from typing import Tuple


def init_weights(l):
        if isinstance(l, nn.Linear):
            torch.nn.init.kaiming_normal_(l.weight, mode='fan_in', nonlinearity='relu')


class MLPNet(nn.Module):
    """
    MLP that takes as input an image embedding and predicts environment light Spherical Harmonics (SH) coefficients
    and sky color SH. The network is trained jointly with the Gaussians.
    
    Attributes:
        sh_dim_envl (int): number of environment lighting SH coefficients.
        sh_dim_sky (int): number of sky color SH coefficients.
        embedding_dim (int): image embedding dimension.
        dense_layer_size (int): size of dense hidden layers.
    """
    def __init__(self,
                 sh_degree_envl: int = 4,
                 sh_degree_sky: int =1,
                 embedding_dim: int = 32,
                 dense_layer_size: int = 256
                 ):
        super().__init__()
        self.sh_dim_envl = (sh_degree_envl + 1) ** 2
        self.sh_dim_sky = (sh_degree_sky + 1) ** 2
        self.embedding_dim = embedding_dim
        self.dense_layer_size = dense_layer_size
        self.optimizer = torch.optim.Adam

        self.base = nn.Sequential(
                nn.Linear(self.embedding_dim, self.dense_layer_size),
                nn.Dropout(p=0.2),
                nn.ReLU(), 
                nn.Linear(self.dense_layer_size, self.dense_layer_size),
                nn.ReLU(),
                nn.Linear(self.dense_layer_size, self.dense_layer_size // 2),
                nn.ReLU(),
            )
            
        self.sh_sky_out = nn.Linear(self.dense_layer_size // 2, self.sh_dim_sky * 3)

        self.sh_envl_head = nn.Sequential(
                                         nn.Linear(self.dense_layer_size // 2,self.dense_layer_size // 2),
                                         nn.ReLU()
                                         )
        self.sh_envl_out = nn.Linear(self.dense_layer_size // 2, self.sh_dim_envl * 3)


    def forward(self, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base_features = self.base(e)
        
        sh_sky = self.sh_sky_out(base_features).view(-1, self.sh_dim_sky, 3)

        sh_envl = self.sh_envl_head(base_features)
        sh_envl = self.sh_envl_out(sh_envl).view(-1, self.sh_dim_envl, 3)       

        return sh_envl, sh_sky


    def get_optimizer(self):
        return self.optimizer(self.parameters(), lr=0.002)
    

    def save_weights(self, path: str, epoch: int):
        torch.save(self.state_dict(), path + "/MLPNet_epoch_" + str(epoch)+ ".pth")
