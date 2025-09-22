# -*- coding: gbk -*-

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch.distributions import normal
import torch.nn.functional as F
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from io_utils import parse_args
from tqdm import tqdm


def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)
    
class VAE(nn.Module):

    def __init__(self, in_channels, latent_dim, hidden_dim):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, in_channels),
            nn.BatchNorm1d(in_channels),
            #nn.Sigmoid()
        )
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
              if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.normal_(0, 0.02)
                

    def encode(self, input):
        result = self.encoder(input)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

    def decode(self, z):

        z = self.decoder_input(z)
        z_out = self.decoder(z)
        return z_out

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        z_out = self.decode(z)
        return z_out, input, mu, log_var

    def loss_function(self, input, rec, mu, log_var, kld_weight=0.00025):
        recons_loss = F.mse_loss(rec, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 +
                              log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        
        loss = recons_loss + kld_weight * kld_loss 

        return {'loss_vae': loss, 'loss_rec': recons_loss, 'loss_kl': kld_loss}
        
           
def train_VAE(feature_loader, feats_VAE, num_epochs, lr=0.001):
    optimizer = torch.optim.Adam(feats_VAE.parameters(), lr=lr)
    for epoch in range(num_epochs):
        loss_recon_all = 0
        loss_kl_all = 0
        for idx, (x, y) in tqdm(enumerate(feature_loader)):
            x = x.cuda().view(-1, x_dim)
            #labels = one_hot(y, num_class).cuda()
            recon_feats, _, mu, log_var = feats_VAE(x)
            
            loss_vae = feats_VAE.loss_function(
            x, recon_feats, mu, log_var)
            
            
            optimizer.zero_grad()
            loss_vae['loss_vae'].backward()   
            optimizer.step()
            
            loss_recon_all += loss_vae['loss_rec'].item()
            loss_kl_all += loss_vae['loss_kl'].item()
            
        print('Ep: %d   Recon Loss: %f   KL Loss: %f'%(epoch, loss_recon_all/(idx+1), loss_kl_all/(idx+1) ))
    feats_VAE.eval()
    return feats_VAE
         
class FeatureDataset(Dataset):
    
    def __init__(self, features, labels):
        super(FeatureDataset, self).__init__()
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def get_hub_samples(z_size, hub_size, dim, t, batch_size = 10):
    
    z_dim = dim
    Z = np.empty((0, z_dim))
    
    while len(Z) < hub_size:
        z = VAE_model.decode(concat_feats).cpu().detach().numpy()
        s = np.zeros(z_size)
        for i in range(0, z_size, batch_size):
            zi = z[i: min(i + batch_size, z_size)]
            d = (z**2).sum(1)[:, None] + (zi**2).sum(1)[None] - 2 * z.dot(zi.T)
            for j in d.argsort(0)[1:1 + 5].T:
                s[j] += 1
        z = z[s > t]
        Z = np.concatenate([Z, z], 0)[:hub_size]
        #print('%s / %s' % (len(Z), hub_size))
    return Z
    

def select_base_samples(features, z_size, hub_rate, batch_size = 10):
    
    hub_size = int(z_size*hub_rate)
    z = features
    s = np.zeros(z_size)
    for i in range(0, z_size, batch_size):
            zi = z[i: min(i + batch_size, z_size)]
            d = (z**2).sum(1)[:, None] + (zi**2).sum(1)[None] - 2 * z.dot(zi.T)
            for j in d.argsort(0)[1:1 + 5].T:
                s[j] += 1
           
    k = int(z_size * hub_rate)
    
    top_indices = np.argpartition(s, -k)[-k:]
    return z[top_indices]
 

if __name__ == '__main__':          
    params = parse_args('test')
    n_ways = params.way
    #n_shot = params.shot
    n_shot = 1
    n_queries = 15
    n_runs = params.run
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples
    
    split = params.split
    method = params.method
    dataset = params.dataset
    dataset = 'miniImagenet'
    num_sampled = 150   
 
    n_runs = 10000
    # --- VAE
    x_dim = 640
    #x_dim=512
    h_dim = 1024
    z_dim = 512
    batch_size = 128 
    num_epochs = 60
    lr = 0.001
    
    path = './feats_vae_mini.pth'
    base_mean = []
    base_cov = []
    
    
    features = []
    labels = []
    
    
    #ver_cov_tensor1 = []
    base_features_path = "./checkpoints/%s/%s/last/base_features.plk"%(params.dataset, params.method)
    with open(base_features_path, 'rb') as f:
        data = pickle.load(f)
        num_class = len(data.keys())
        condition_dim = num_class
        for idx, key in enumerate(data.keys()):
            feature = np.array(data[key])
            mean = np.mean(feature, axis=0)
            #cov = np.cov(feature.T)
            if not os.path.exists(path):
                feature = select_base_samples(features = feature, z_size = len(feature), hub_rate = 0.2, batch_size = 10)
            
                features.append(feature)
                labels.append([idx]*len(feature))
            
            #mean = np.mean(feature, axis=0)
            base_mean.append(mean)
            #base_cov.append(cov)
          
    #base_mean = torch.Tensor(np.array(base_mean)).cuda()
    #base_mean = F.normalize(base_mean) 
    #base_cov = torch.Tensor(np.array(base_cov)).cuda() 
    
    if os.path.exists(path):   
        VAE_model = torch.load(path)
    else:
        feat_dataset = FeatureDataset(np.vstack(features), np.hstack(labels))
        data_loader = DataLoader(dataset=feat_dataset, shuffle=True, pin_memory=True, drop_last=False, batch_size=batch_size)
    
        VAE_model = VAE(in_channels=x_dim, latent_dim=z_dim, hidden_dim=h_dim).cuda()
        VAE_model = train_VAE(data_loader, VAE_model, num_epochs, lr = lr)
        torch.save(VAE_model, path)
    
    
    import FSLTask
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries, 'method': method, 'split':split}

    FSLTask.loadDataSet(dataset, method, split)
    FSLTask.setRandomStates(cfg)       
    ndatas = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg)    
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, n_ways).clone().view(n_runs,
                                                                                                        n_samples)                                                                         
                                                                                                        
    
    z_dist = normal.Normal(0, 1)
           
    acc_list = []
    for it in tqdm(range(n_runs)):
        # ---support samples
        support_data = ndatas[it][:n_lsamples].numpy()
        support_label = labels[it][:n_lsamples].numpy()
        
        # ---query samples
        query_data = ndatas[it][n_lsamples:n_lsamples+n_usamples].numpy()
        query_label = labels[it][n_lsamples:n_lsamples+n_usamples].numpy()
        
        #beta = 0.75
        #support_data = np.power(support_data[:, ] ,beta)
        #query_data = np.power(query_data[:, ] ,beta)
        
        # --- Calculate the prototype
        support_set = [support_data[j::n_ways] for j in range(n_ways)]
        prototype = [np.mean(support_set[j], axis=0) for j in range(n_ways)]
        
        X_aug = []
        Y_aug = []
        
        base_mean = np.array(base_mean)
        prototype = np.array(prototype)
        #dist = (prototype**2).sum(1)[:, None] + (base_mean**2).sum(1)[None] - 2 * prototype.dot(base_mean.T)
        #dist = dist**0.5
        
        for i in range(n_ways):    
            
            Z = z_dist.sample((num_sampled, z_dim)).cuda()
            
            #diversity = get_hub_samples(z_size = num_sampled, hub_size = num_selected, dim = dim, t = 5, batch_size = 10)
            
            t = VAE_model.decode(Z).cpu().detach().numpy()
            
            #t = select_base_samples(features = t, z_size = num_sampled, hub_rate = 0.3, batch_size = 10)
            
            #t = diversity
            num_selected = len(t)
           
            
            alpha = 0.6
            for j in range(num_selected):
                direction = prototype[i] - t[j]
                t[j] += alpha * direction
            
            X_aug.append(t)
            Y_aug.extend([support_label[i]]*num_selected)
        
     
        X_aug = np.concatenate([X_aug[:]]).reshape(n_ways * num_selected, -1)
        
        
    
        X_aug = np.concatenate([X_aug, support_data])
        Y_aug = np.concatenate([Y_aug, support_label])
        
        X_aug[X_aug < 0] = 0
        beta = 0.5
        #support_data = np.power(support_data[:, ] ,beta)
        query_data = np.power(query_data[:, ] ,beta)
        X_aug = np.power(X_aug[:, ] ,beta)
        
        X_aug = F.normalize( torch.tensor(np.array(X_aug))).numpy()
        query_data = F.normalize( torch.tensor(np.array(query_data))).numpy()
        #tsne(X_aug,Y_aug)
        #break
        # --- rain a new classifier
        classifier = LogisticRegression(max_iter=1000).fit(X=X_aug, y=Y_aug)
        #classifier = LogisticRegression(max_iter=1000).fit(X=support_data, y=support_label) #baseline
      
        # --- Predict the labels of query samples
        predicts = classifier.predict(query_data)
        acc = np.mean(predicts == query_label)
        acc_list.append(acc*100)
        print(it,acc)
        if (it+1)%10 == 0:
            sqr_n = np.sqrt(it+1)
            print(it+1, params.method, '%s %d way %d shot  ACC: %4.2f%% +- %4.2f%%'%(params.dataset, n_ways, n_shot,
                      float(np.mean(acc_list)),1.96*np.std(acc_list)/sqr_n))