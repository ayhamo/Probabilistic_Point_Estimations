from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import os
import time
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.autograd import Variable
from .util import DataTransformer, reparameterize, _loss_function_MMD,z_gen
# from ttvae.base import BaseSynthesizer, random_state

from torch.distributions import Normal
import properscoring as ps



class Encoder_T(nn.Module):
    def __init__(self, input_dim, latent_dim, embedding_dim, nhead, dim_feedforward=2048, dropout=0.1):
      super(Encoder_T, self).__init__()
      # Input data to Transformer
      self.linear = nn.Linear(input_dim,embedding_dim)
      # Transformer Encoder
      self.transformerencoder_layer = nn.TransformerEncoderLayer(embedding_dim, nhead, dim_feedforward, dropout, batch_first=True)
      self.encoder = nn.TransformerEncoder(self.transformerencoder_layer, num_layers=2)
      # Latent Space Representation
      self.fc_mu = nn.Linear(embedding_dim, latent_dim)
      self.fc_log_var = nn.Linear(embedding_dim, latent_dim)

    def forward(self, x):
      # Encoder
      x = self.linear(x)
      enc_output = self.encoder(x)
      # Latent Space Representation
      mu = self.fc_mu(enc_output)
      logvar = self.fc_log_var(enc_output)
      std = torch.exp(0.5 * logvar)
      return mu, std, logvar, enc_output


class Decoder_T(nn.Module):
    def __init__(self, input_dim, latent_dim, embedding_dim, nhead, dim_feedforward=2048, dropout=0.1):
      super(Decoder_T, self).__init__()
      # Linear layer for mapping latent space to decoder input size
      self.latent_to_decoder_input = nn.Linear(latent_dim, embedding_dim)
      # Transformer Decoder
      self.transformerdecoder_layer = nn.TransformerDecoderLayer(embedding_dim, nhead, dim_feedforward, dropout)
      self.decoder = nn.TransformerDecoder(self.transformerdecoder_layer, num_layers=2)
      # Transformer Embedding to input
      self.linear = nn.Linear(embedding_dim,input_dim)
      self.sigma = Parameter(torch.ones(input_dim) * 0.1)

    def forward(self, z, enc_output):
      # Encoder
      z_decoder_input = self.latent_to_decoder_input(z)
      # Decoder
      # Note: Pass enc_output (memory) to the decoder
      dec_output = self.decoder(z_decoder_input, enc_output)

      return self.linear(dec_output), self.sigma


class TTVAE():
    """TTVAE."""

    def __init__(
        self,
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2,
        latent_dim =32,# Example latent dimension
        embedding_dim=128,# Transformer embedding dimension
        nhead=8,# Number of attention heads
        dim_feedforward=1028,# Feedforward layer dimension
        dropout=0.1,
        cuda=True,
        verbose=False,
        device='cuda'
    ):
        self.latent_dim=latent_dim
        self.embedding_dim = embedding_dim
        self.nhead=nhead
        self.dim_feedforward=dim_feedforward
        self.dropout=dropout
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs
        self._device = torch.device(device)

    # @random_state
    def fit(self, train_data, discrete_columns=(),save_path=''):
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)

        self.train_data = self.transformer.transform(train_data).astype('float32')
        dataset = TensorDataset(torch.from_numpy(self.train_data).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        data_dim = self.transformer.output_dimensions

        print(f"data_dim: {data_dim}, latent_dim: {self.latent_dim}, embedding_dim: {self.embedding_dim}, "
        f"nhead: {self.nhead}, dim_feedforward: {self.dim_feedforward}, dropout: {self.dropout}")


        self.encoder = Encoder_T(data_dim, self.latent_dim, self.embedding_dim, self.nhead, self.dim_feedforward, self.dropout).to(self._device)
        self.decoder = Decoder_T(data_dim, self.latent_dim, self.embedding_dim, self.nhead, self.dim_feedforward, self.dropout).to(self._device)

        optimizer = Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), weight_decay=self.l2scale)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20)

        self.encoder.train()
        self.decoder.train()

        best_loss = float('inf')
        patience = 0
        start_time = time.time()

        for epoch in range(self.epochs):
        
            #pbar = tqdm(enumerate(loader), total=len(loader))
            #pbar.set_description(f"Epoch {epoch+1}/{self.epochs}")

            batch_loss = 0.0
            len_input = 0

            #for id_, data in pbar:
            for id_, data in enumerate(loader):
                optimizer.zero_grad()
                real_x = data[0].to(self._device)
                mean, std, logvar, enc_output = self.encoder(real_x)
                z = reparameterize(mean, logvar)
                recon_x, sigmas = self.decoder(z,enc_output)
                loss = _loss_function_MMD(recon_x, real_x, sigmas, mean, logvar, self.transformer.output_info_list, self.loss_factor)

                batch_loss += loss.item() * len(real_x)
                len_input += len(real_x)

                loss.backward()
                optimizer.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

                #pbar.set_postfix({"Loss": loss.item()})

            curr_loss = batch_loss/len_input
            scheduler.step(curr_loss)

            if curr_loss < best_loss:
              best_loss = loss.item()
              patience = 0
              torch.save(self, save_path+'/model.pt')
            else:
                patience += 1
                if patience == 500:
                    print('Early stopping')
                    break

                        
    # @random_state
    def sample(self, n_samples=100):
        """Sample data similar to the training data.

        """
        self.encoder.eval()
        with torch.no_grad():
            mean, std, logvar, enc_embed= self.encoder(torch.Tensor(self.train_data).to(self._device))

        embeddings = torch.normal(mean=mean, std=std).cpu().detach().numpy()
        synthetic_embeddings=z_gen(embeddings,n_to_sample=n_samples,metric='minkowski',interpolation_method='SMOTE')
        noise = torch.Tensor(synthetic_embeddings).to(self._device)

        self.decoder.eval()
        with torch.no_grad():
          fake, sigmas = self.decoder(noise,enc_embed)
          fake = torch.tanh(fake).cpu().detach().numpy()

        return self.transformer.inverse_transform(fake)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        self.decoder.to(self._device)

    
    def predict_distribution(self, data_df, target_col_name, n_samples=100):
        """
        Generates a distribution of samples for the target column for each row in the input dataframe.
        This version is optimized for performance by vectorizing the inverse transform.
        """
        self.encoder.eval()
        self.decoder.eval()

        # --- Step 1: Transform data and create loader ---
        transformed_data = self.transformer.transform(data_df).astype('float32')
        data_tensor = torch.from_numpy(transformed_data).to(self._device)
        
        all_recon_tensors = []
        with torch.no_grad():
            loader = DataLoader(TensorDataset(data_tensor), batch_size=self.batch_size, shuffle=False)
            for batch in loader:
                real_x = batch[0]
                # --- Step 2: Encode, sample in latent space, and decode ---
                mean, std, _, enc_output = self.encoder(real_x)
                
                expanded_mean = mean.unsqueeze(1).expand(-1, n_samples, -1)
                expanded_std = std.unsqueeze(1).expand(-1, n_samples, -1)
                z_samples = torch.normal(mean=expanded_mean, std=expanded_std)
                
                expanded_enc_output = enc_output.unsqueeze(1).expand(-1, n_samples, -1)
                
                batch_size, _, latent_dim = z_samples.shape
                z_flat = z_samples.reshape(-1, latent_dim)
                enc_output_flat = expanded_enc_output.reshape(-1, enc_output.shape[-1])
                
                recon_flat, _ = self.decoder(z_flat, enc_output_flat)
                
                recon_samples = recon_flat.reshape(batch_size, n_samples, -1)
                
                # --- Step 3: Collect raw tensors from the GPU ---
                all_recon_tensors.append(recon_samples)

        # --- Step 4: Post-process ALL samples at once outside the loop ---
        
        # Concatenate all batch results into one big tensor on the GPU
        # Shape: [total_num_rows, n_samples, num_features]
        final_recon_tensor = torch.cat(all_recon_tensors, dim=0)

        # Apply final activation function (e.g., tanh)
        # This was previously applied in the old function, so we keep it for consistency.
        final_recon_tensor = torch.tanh(final_recon_tensor)
        
        # Reshape for a single inverse_transform call.
        # We flatten the first two dimensions (rows and samples).
        # Shape becomes: [total_num_rows * n_samples, num_features]
        num_rows, _, num_features = final_recon_tensor.shape
        flat_recon_tensor = final_recon_tensor.reshape(-1, num_features)
        
        flat_recon_np = flat_recon_tensor.cpu().numpy()

        # Perform a SINGLE, large inverse transform. This is massively faster.
        inversed_df = self.transformer.inverse_transform(flat_recon_np)
        
        # Extract only the target column we need for CRPS
        target_values = inversed_df[target_col_name].values
        
        # Reshape the 1D array of target values back to the desired 2D shape
        # Shape: [total_num_rows, n_samples]
        final_distributions = target_values.reshape(num_rows, n_samples)
        
        return final_distributions
    
    def estimate_crps(self, test_df, target_col_name):
        
        # 1. Generate the predictive distributions
        # This now returns a NumPy array of shape [num_test_rows, 100]
        distributions = self.predict_distribution(test_df, target_col_name, n_samples=100)

        # 2. Get the true target values
        y_true = test_df[target_col_name].values

        total_crps = 0.0
        for i in range(len(test_df)):
            # Get the forecast distribution (it's already a NumPy array)
            y_forecast_dist = distributions[i]
            
            # Get the true value
            y_observed = y_true[i]
            
            # Calculate CRPS for this one sample
            crps_score = ps.crps_ensemble(y_observed, y_forecast_dist)
            total_crps += crps_score

        average_crps = total_crps / len(test_df)
        return average_crps
    
    
    def _get_log_prob(self, real_x, recon_x, sigmas):
        """Helper to calculate the log probability of the reconstruction."""
        log_likelihood = 0
        current_pos = 0
        
        # The outer loop iterates through the original columns.
        # 'column_info_list' is a list like [SpanInfo(...)] or [SpanInfo(...), SpanInfo(...)]
        for column_info_list in self.transformer.output_info_list:
            
            # The inner loop iterates through the components of that column's transformation.
            # 'info' is now the actual SpanInfo object.
            for info in column_info_list:
                dim = info.dim
                
                # Select the slice of data for this component
                data_slice = real_x[:, current_pos : current_pos + dim]
                recon_slice = recon_x[:, current_pos : current_pos + dim]
                
                if info.activation_fn == 'tanh':
                    # This is a numerical component (from a GMM or a simple numerical feature)
                    # We use the overall decoder sigma for the Gaussian noise
                    sigma_slice = self.decoder.sigma[current_pos : current_pos + dim]
                    
                    # Calculate the log probability under a Gaussian distribution
                    log_likelihood += Normal(loc=recon_slice, scale=sigma_slice).log_prob(data_slice).sum()

                else: # 'softmax'
                    # This is a categorical component (from one-hot encoding or a GMM cluster choice)
                    # The log probability for a categorical distribution is the cross-entropy
                    log_likelihood += -(torch.log_softmax(recon_slice, dim=-1) * data_slice).sum()
                
                current_pos += dim
                
        return log_likelihood
    
def estimate_nll_target(self, data_df, n_samples=500, sample_chunk_size=100):
    """
    Estimate NLL of the last column (target y) given all other columns (features X).
    """
    self.encoder.eval()
    self.decoder.eval()

    # Split features (all but last col) and target (last col)
    X = data_df.iloc[:, :-1].values.astype('float32')
    y = data_df.iloc[:, -1].values.astype('float32')

    X_tensor = torch.from_numpy(X).to(self._device)
    y_tensor = torch.from_numpy(y).to(self._device)

    prior = Normal(loc=torch.zeros(self.latent_dim, device=self._device),
                   scale=torch.ones(self.latent_dim, device=self._device))

    total_nll = 0.0
    with torch.no_grad():
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=False)
        for batch_X, batch_y in loader:
            # Proposal q(z|X,y)
            mean_q, std_q, _, enc_output = self.encoder(batch_X, batch_y)
            q_dist = Normal(loc=mean_q, scale=std_q)

            all_log_weights = []
            num_chunks = (n_samples + sample_chunk_size - 1) // sample_chunk_size

            for _ in range(num_chunks):
                # Sample z from q(z|X,y)
                z_samples = q_dist.rsample(sample_shape=(sample_chunk_size,))
                z_flat = z_samples.view(-1, self.latent_dim)

                enc_out_expanded = enc_output.repeat(sample_chunk_size, 1)
                batch_y_expanded = batch_y.repeat(sample_chunk_size, 1).view(-1)

                # Decoder predicts parameters of p(y|X,z)
                mu_y, sigma_y = self.decoder(z_flat, enc_out_expanded)

                # Likelihood of target only
                log_p_y_xz = Normal(loc=mu_y, scale=sigma_y).log_prob(batch_y_expanded)

                log_p_z = prior.log_prob(z_flat).sum(dim=-1)
                log_q_z_xy = q_dist.log_prob(z_samples).sum(dim=-1).view(-1)

                log_weights_chunk = log_p_y_xz + log_p_z - log_q_z_xy
                all_log_weights.append(log_weights_chunk)

            # Combine chunks
            log_weights = torch.cat(all_log_weights).view(n_samples, batch_X.shape[0]).T
            log_likelihood = torch.logsumexp(log_weights, dim=1) - torch.log(
                torch.tensor(n_samples, dtype=torch.float, device=self._device)
            )

            total_nll -= log_likelihood.sum().item()

    return total_nll / len(data_df)


