import torch
import torch.nn as nn
from torch.nn import functional as F

class conditional_VAE(nn.Module):
    def __init__(self,in_dims: int,
                 num_cond: int,
                 latent_dim: int,
                 hidden_dims: list = None,):
        super(conditional_VAE, self).__init__()
        self.in_dims = in_dims
        self.latent_dim = latent_dim
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]
        in_dims += 1 # To account for the extra label channel

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(nn.Linear(in_dims, h_dim),
                nn.LeakyReLU())
            )
            in_dims = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim + num_cond, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(nn.Linear(hidden_dims[i],
                                       hidden_dims[i + 1]),
                nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.Linear(hidden_dims[-1],self.in_dims),
                            nn.Softmax(dim=-1))
        self.sigmoid_layer = nn.Sequential(nn.Linear(hidden_dims[-1],self.in_dims),
                            nn.Sigmoid())
                        
    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    def decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        result1 = self.final_layer(result)
        sigmoid_result = self.sigmoid_layer(result)
        return result1, sigmoid_result
    
    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, cond, **kwargs):
        x = torch.cat([input, cond], dim = 1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z = torch.cat([z, cond], dim = 1)
        result, sigmoid = self.decode(z)
        return  [result, sigmoid, input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        sigmoid = args[1]
        input = args[2]
        mu = args[3]
        log_var = args[4]

        kld_weight = 1
        recons_loss = nn.MSELoss()(recons,input) # Account for the minibatch samples from the dataset
        #recons_loss =(input-recons)**2
        #weights = torch.ones_like(input) - input
        #weighted_recons_loss = torch.mean(weights*recons_loss)

        test = (input> 0).float()
        mean = torch.sum(test, axis=1).reshape(-1,1)
        test /= mean

        special_loss = nn.CrossEntropyLoss()(recons,test)

        #target = (input> 0).float()
        #sigmoid_loss = nn.BCELoss()(sigmoid, target)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = kld_weight * kld_loss + special_loss*recons_loss #+ sigmoid_loss
        return loss, recons_loss, kld_loss
        #return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}
    
    def sample(self,
               num_samples:int,
               cond,
               current_device: int,
               **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        z = torch.cat([z, cond], dim=1)
        samples = self.decode(z)
        return samples

    def generate(self, x,cond, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x,cond, **kwargs)[0]

class MLP_torch(nn.Module):
    def __init__(self,in_size,  hidden_size, out_size):
        super(MLP_torch,self).__init__()
        self.fc1 = nn.Linear(in_size,hidden_size)
        self.mapf = nn.Linear(hidden_size,out_size)
        self.act = nn.ELU()
    
    def forward(self,x):
        x = self.act(self.fc1(x))
        return self.mapf(x)

def BCELoss_class_weighted(input, target):
    weights = torch.ones_like(input) - input
    #weights = torch.clamp(input,min=1e-7,max=1-1e-7)
    bce = - weights * target * torch.log(input) - (1 - target) * weights * torch.log(1 - input)
    return torch.mean(bce)

class Generator(nn.Module):
    def __init__(self, in_size, n_hidden_layers, hidden_size, out_size):
        super(Generator,self).__init__()
    
        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
    
        self.model = nn.Sequential(*block(in_size,hidden_size),
        *n_hidden_layers*block(hidden_size,hidden_size),
        nn.Linear(hidden_size,out_size),
        nn.Softmax(dim=-1))
    
    def forward(self,noise,prop):
        input = torch.cat((noise,prop),-1)
        x = self.model(input)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_size, n_hidden_layers, hidden_size, out_size):
        super(Discriminator,self).__init__()

        def block(in_feat,out_feat):
            layers = [nn.Linear(in_feat,out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
    
        self.model = nn.Sequential(*block(in_size,hidden_size),
        *n_hidden_layers*block(hidden_size,hidden_size),
        nn.Linear(hidden_size,out_size),
        nn.Sigmoid())
    
    def forward(self,comp,prop):
        input = torch.cat((comp,prop),-1)
        x = self.model(input)
        return x
