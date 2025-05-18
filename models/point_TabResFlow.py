from configs.logger_config import global_logger as logger
from utils.evaluation import evaluate_nll 

import copy
import os
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
import zuko

# Numerical Embedding MLP
class NumericalFeatureEncoder(nn.Module):
    def __init__(self, output_embedding_dim: int, intermediate_dim: int = 100):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, output_embedding_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

# ResNet Block Helper
class ResNetBlock(nn.Module):
    def __init__(self, 
                 input_dim: int, # This is 'd' (resnet_main_processing_dim)
                 hidden_layer_multiplier: float = 1.0,
                 activation_dropout_rate: float = 0.1,
                 residual_dropout_rate: float = 0.1):
        super().__init__()
        actual_hidden_dim = int(input_dim * hidden_layer_multiplier) 
        
        self.bn = nn.BatchNorm1d(input_dim)
        self.linear1 = nn.Linear(input_dim, actual_hidden_dim)

        self.activation = nn.ReLU() 
        self.dropout_after_activation = nn.Dropout(activation_dropout_rate)
        self.linear2 = nn.Linear(actual_hidden_dim, input_dim)
        self.dropout_before_residual_add = nn.Dropout(residual_dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn(x)
        out = self.linear1(out) # Corrected: operate on 'out' from bn
        out = self.activation(out)
        out = self.dropout_after_activation(out)
        out = self.linear2(out)
        out = self.dropout_before_residual_add(out)
        out += identity 
        return out

class TabResFlow(nn.Module):
    def __init__(
        self,
        num_numerical_features: int,
        
        # Numerical Feature Embedding Specific:
        embedding_dim_per_feature: int = 64,     # Author 'dim' for ResNetModel individual feature embeddings
        numerical_encoder_intermediate_dim: int = 100, # For the 2-layer numerical MLP encoder

        # ResNet Backbone Specific:
        resnet_main_processing_dim: int = 256, # Author 'hidden_dim' (d) in ResNetModel; context dim for flow
        resnet_depth: int = 4,                 # Author 'depth' (number of ResNet blocks)
        resnet_block_hidden_factor: float = 1.0, # Author 'd_hidden_factor' for MLP within ResNet block
                                                 # Multiplies resnet_main_processing_dim for hidden layer in block
        resnet_activation_dropout: float = 0.1,  # Author 'hidden_dropout' (after activation in block)
        resnet_residual_dropout: float = 0.1,    # Author 'residual_dropout' (before adding residual)
        
        # Normalizing Flow Head Specific:
        flow_transforms: int = 3,                       # Author 'flow_num_blocks' for NSF
        flow_mlp_layers_in_transform: int = 3,          # Author 'flow_layers' (num hidden layers in each NSF transform's MLP)
        # flow_mlp_hidden_features_in_transform is implicitly resnet_main_processing_dim (see NSF init below)
        flow_bins: int = 8,                           
        
        # Categorical params (currently unused as UCI is numerical)
        categorical_cardinalities = None, 
        category_embedding_dim: int = 64, # Usually same as embedding_dim_per_feature
        target_scaler_actual_scale = None,
    ):
        super().__init__()
        self.num_numerical = num_numerical_features
        self.num_categorical = len(categorical_cardinalities) if categorical_cardinalities else 0

        # 1. Numerical Feature Encoders
        self.numerical_encoders = nn.ModuleList([
            NumericalFeatureEncoder(
                output_embedding_dim=embedding_dim_per_feature,
                intermediate_dim=numerical_encoder_intermediate_dim
            ) for _ in range(self.num_numerical)
        ])

        # 2. Categorical Feature Embeddings (if any)
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=card, embedding_dim=category_embedding_dim)
            for card in (categorical_cardinalities if categorical_cardinalities else [])
        ])

        # 3. ResNet Backbone Construction
        total_concatenated_embedding_dim = (self.num_numerical * embedding_dim_per_feature) + \
                                     (self.num_categorical * category_embedding_dim)
        
        if total_concatenated_embedding_dim == 0:
            raise ValueError("Model initialized with no features, leading to zero embedding dimension.")

        # Initial projection layer
        self.input_projection = nn.Linear(total_concatenated_embedding_dim, resnet_main_processing_dim)
        
        # ResNet Blocks
        self.resnet_blocks = nn.ModuleList([
            ResNetBlock(
                input_dim=resnet_main_processing_dim,
                hidden_layer_multiplier=resnet_block_hidden_factor,
                activation_dropout_rate=resnet_activation_dropout,
                residual_dropout_rate=resnet_residual_dropout
            ) for _ in range(resnet_depth)
        ])

        # 4. Conditional Normalizing Flow Head
        # Aligning with author NSF initialization:
        # hidden_features for NSF transform's MLP = [context_dim] * num_layers_in_transform_mlp
        nsf_transform_mlp_hidden_features = [resnet_main_processing_dim] * flow_mlp_layers_in_transform

        self.normalizing_flow = zuko.flows.NSF(
            features=1, # Univariate target                          
            context=resnet_main_processing_dim, 
            transforms=flow_transforms,
            bins=flow_bins,
            hidden_features=nsf_transform_mlp_hidden_features, 
            randperm=False, 
        )

        self.target_scaler_actual_scale = target_scaler_actual_scale

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor):
        num_embeds = []
        if self.num_numerical > 0 and x_num.numel() > 0:
            # x_num shape: (batch_size, num_numerical_features)
            for i in range(self.num_numerical):
                feat = x_num[:, i:i+1] # Input to NumericalFeatureEncoder is (batch_size, 1)
                embed = self.numerical_encoders[i](feat) # Output: (batch_size, embedding_dim_per_feature)
                num_embeds.append(embed)

        cat_embeds = []
        if self.num_categorical > 0 and x_cat.numel() > 0:
            for i, emb_layer in enumerate(self.categorical_embeddings):
                cat_embeds.append(emb_layer(x_cat[:, i])) # x_cat[:, i] is (batch_size,)
        
        all_embeds = []

        if num_embeds: all_embeds.extend(num_embeds) # List of tensors
        if cat_embeds: all_embeds.extend(cat_embeds) # List of tensors
        
        if not all_embeds:
            batch_s = x_num.shape[0] if x_num.numel() > 0 else (x_cat.shape[0] if x_cat.numel() > 0 else 0)
            if batch_s == 0:
                 return torch.empty(0, self.normalizing_flow.context_size, device=next(self.parameters()).device)
            # This case should ideally be caught by __init__ if no features are defined at all.
            logger.error("No embeddings generated in forward pass for a non-empty batch. Check model init and input data.")
            # Fallback to zero context, but this is problematic.
            return torch.zeros(batch_s, self.normalizing_flow.context_size, device=next(self.parameters()).device)

        combined_embeddings = torch.cat(all_embeds, dim=1) # Concatenate all feature embeddings
        
        z = self.input_projection(combined_embeddings)
        for block in self.resnet_blocks:
            z = block(z)
        return z

    def log_prob(self, y: torch.Tensor, x_num: torch.Tensor, x_cat: torch.Tensor):
        if y.numel() == 0: return torch.empty(0, device=y.device)
        context = self.forward(x_num, x_cat)
        if context.shape[0] == 0: return torch.empty(0, device=context.device)
        
        distribution = self.normalizing_flow(context)
        log_p_y_scaled = distribution.log_prob(y) # y is the scaled target batch

        # correction is essential when the target variable has been scaled—ensuring that the computed negative log-likelihood (NLL) refers to the original scale.
        log_s_eff = torch.log(torch.tensor(self.target_scaler_actual_scale, device=y.device, dtype=y.dtype))
        log_p_original_scale = log_p_y_scaled + log_s_eff
        return log_p_original_scale
 

    def sample(self, x_num: torch.Tensor, x_cat: torch.Tensor, num_samples: int = 100):
        if x_num.numel() == 0 and x_cat.numel() == 0 and self.num_numerical == 0 and self.num_categorical == 0 : # Truly no input and no features defined
            # This case should ideally be prevented by __init__ checks for num_features > 0
            return torch.empty(0, num_samples, 1, device=next(self.parameters()).device)
        
        context = self.forward(x_num, x_cat)
        if context.shape[0] == 0: # If forward returned empty context for an empty batch input
             return torch.empty(0, num_samples, 1, device=context.device)
             
        distribution = self.normalizing_flow(context)
        # rsample returns (S, B, D) where S=num_samples, B=batch_size, D=feature_dim (1)
        return distribution.rsample((num_samples,)) 

    def predict_mean_std(self, x_num: torch.Tensor, x_cat: torch.Tensor, num_mc_samples: int = 500):
        self.eval() 
        # Handle case where input batch might be empty (e.g., last batch from DataLoader)
        current_batch_size = x_num.shape[0] if self.num_numerical > 0 and x_num.numel() > 0 else (x_cat.shape[0] if self.num_categorical > 0 and x_cat.numel() > 0 else 0)
        if current_batch_size == 0:
             # Get device from a parameter if possible, fallback to cpu
            model_device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')
            return torch.empty(0, device=model_device), torch.empty(0, device=model_device)


        with torch.no_grad():
            samples = self.sample(x_num, x_cat, num_samples=num_mc_samples) # Shape: (num_mc_samples, batch_size, 1)
            
            # Check if sampling itself returned empty due to an empty context from forward()
            if samples.numel() == 0 or samples.shape[1] == 0: # samples.shape[1] is batch_size
                model_device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')
                return torch.empty(0, device=model_device), torch.empty(0, device=model_device)
            
            mean = samples.mean(dim=0) # Shape: (batch_size, 1)
            std = samples.std(dim=0)   # Shape: (batch_size, 1)
        return mean.squeeze(-1), std.squeeze(-1) # Shape: (batch_size,)
    
    def predict_samples_original_scale(
        self, 
        x_num: torch.Tensor, 
        x_cat: torch.Tensor, 
        target_scaler: MinMaxScaler,
        num_mc_samples: int = 1000
    ):
        """
        Generates samples from the predictive distribution and returns them
        on the original data scale.
        Output shape: (batch_size, num_mc_samples)
        """
        self.eval()
        current_batch_size = x_num.shape[0] if self.num_numerical > 0 and x_num.numel() > 0 else \
                                (x_cat.shape[0] if self.num_categorical > 0 and x_cat.numel() > 0 else 0)
        
        if current_batch_size == 0:
            return np.array([]).reshape(0, num_mc_samples) # Return empty array with correct second dim

        with torch.no_grad():
            # self.sample returns (num_mc_samples, batch_size, 1) in scaled space
            samples_scaled = self.sample(x_num, x_cat, num_samples=num_mc_samples)
            
            if samples_scaled.numel() == 0 or samples_scaled.shape[1] == 0: # samples.shape[1] is batch_size
                    return np.array([]).reshape(0, num_mc_samples)

            # Reshape for inverse_transform: (num_mc_samples * batch_size, 1)
            batch_size_actual = samples_scaled.shape[1] # Get actual batch size from samples
            samples_scaled_reshaped = samples_scaled.permute(1, 0, 2).reshape(batch_size_actual * num_mc_samples, 1)
            samples_scaled_np = samples_scaled_reshaped.cpu().numpy()

            if target_scaler is None:
                logger.error("Target scaler is None in predict_samples_original_scale. Cannot inverse transform.")
                # Or raise error, or return scaled samples with a warning
                return samples_scaled.permute(1,0,2).squeeze(-1).cpu().numpy() # (batch_size, num_mc_samples)

            samples_original_np_flat = target_scaler.inverse_transform(samples_scaled_np)
            
            # Reshape back to (batch_size, num_mc_samples)
            samples_original_np = samples_original_np_flat.reshape(batch_size_actual, num_mc_samples)
            
        return samples_original_np

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            device: torch.device,
            lr: float = 1e-4, 
            weight_decay: float = 1e-5, 
            num_epochs: int = 100, 
            patience_early_stopping: int = 15,
            model_save_path = None,
            dataset_key_for_save = None
           ):

        self.to(device)
        optimizer = optim.RAdam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val_loss = float('inf')
        best_model_state_dict = None 
        epochs_no_improve = 0

        logger.info(f"Starting training on {device} for {num_epochs} epochs... LR={lr}, WD={weight_decay}, Optim: RAdam")
        
        for epoch in range(num_epochs):
            self.train()
            total_train_loss = 0.0
            batches_processed_train = 0
            for batch_idx, (x_num_b_loader, y_b_loader) in enumerate(train_loader): # Expecting 2 items
                if y_b_loader.numel() == 0 : continue 
                
                x_num_b = x_num_b_loader.to(device)
                y_b = y_b_loader.to(device)
                x_cat_b = torch.empty((x_num_b.shape[0], 0), dtype=torch.long, device=device)

                optimizer.zero_grad()
                try:
                    log_probs = self.log_prob(y_b, x_num_b, x_cat_b)
                    if log_probs.numel() == 0 : continue 
                    loss = -log_probs.mean()
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"NaN or Inf loss in training: epoch {epoch+1}, batch {batch_idx}. Skipping batch.")
                        continue
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()
                    batches_processed_train +=1
                except Exception as e:
                    logger.error(f"Error during training batch (Epoch {epoch+1}, Batch {batch_idx}): {e}", exc_info=True)
                    continue
            
            avg_train_loss = total_train_loss / batches_processed_train if batches_processed_train > 0 else float('nan')

            avg_val_loss = evaluate_nll(
                model=self, data_loader=val_loader, device=device, current_epoch_num=epoch+1
            )
            
            if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:

                logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                best_val_loss = avg_val_loss
                best_model_state_dict = copy.deepcopy(self.state_dict()) 
                epochs_no_improve = 0

                if model_save_path : 
                    final_save_path = model_save_path
                    if dataset_key_for_save:
                         final_save_path = model_save_path.replace(".pth", f"_{dataset_key_for_save}.pth")
                    save_dir = os.path.dirname(final_save_path)
                    if save_dir and not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
                    torch.save({
                        'epoch': epoch, 'model_state_dict': best_model_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(), 'loss': best_val_loss,
                    }, final_save_path)
                    logger.info(f"Model state also saved to {final_save_path}")
            elif not np.isnan(avg_val_loss): 
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience_early_stopping:
                logger.info(f"Early stopping at epoch {epoch+1} after {patience_early_stopping} epochs with no improvement.")
                break
        
        logger.info(f"Training finished. Best Validation NLL: {best_val_loss:.4f}")
        return best_val_loss, best_model_state_dict