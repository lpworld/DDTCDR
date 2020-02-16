import torch
from utils import use_cuda, resume_checkpoint
    
class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.latent_dim = config['latent_dim']
        self.nlayers = config['nlayers']
        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()
        self.bridge = torch.nn.Linear(config['latent_dim'], config['latent_dim'])
        torch.nn.init.orthogonal_(self.bridge.weight)

    def init_weight(self):
        pass
        
    def forward(self, user_embeddings, item_embeddings, dual=False):
        if dual:
            user_embeddings = self.bridge(user_embeddings)
        vector = torch.cat([user_embeddings, item_embeddings], dim=-1)  # the concat latent vector
        vector = vector.float()
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.Dropout(p=0.1)(vector)
            vector = torch.nn.ReLU()(vector)
            #vector = torch.nn.BatchNorm1d()(vector)
        rating = self.affine_output(vector)
        #rating = self.logistic(rating)
        return rating