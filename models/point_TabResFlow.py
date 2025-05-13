from configs.logger_config import global_logger as logger
from utils.evaluation import evaluate_nll

import copy
import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import zuko


# ResNet Block Helper
class ResNetBlock(nn.Module):
    """
    A standard ResNet block with bn, ReLU, and Dropout
    """
    def __init__(self, input_dim: int, hidden_dim_multiplier: int = 2, dropout_rate: float = 0.2):
        super().__init__()
        hidden_dim = input_dim * hidden_dim_multiplier
        self.bn = nn.BatchNorm1d(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        identity = x
        out = self.bn(x)
        out = self.linear1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.dropout2(out)

        out += identity # Residual connection
        # No final activation mentioned for the block itself in common ResNet patterns referenced
        return out

# Main TabResFlow Model
class TabResFlow(nn.Module):
    """
    Implements the TabResFlow model architecture.

    Args:
        num_numerical_features: Number of numerical input features.
        categorical_cardinalities: List of cardinalities (number of unique values) for each categorical feature.
        numerical_embedding_dim: Output dimension for the numerical feature encoder MLP.
        category_embedding_dim: Embedding dimension for each categorical feature.
        resnet_hidden_dim_multiplier: Multiplier for hidden layer size in ResNet blocks.
        resnet_num_blocks: Number of ResNet blocks.
        resnet_dropout: Dropout rate within ResNet blocks.
        flow_num_transforms: Number of transformation layers in the Normalizing Flow.
        flow_hidden_features: Number of hidden features in the MLPs defining the spline parameters within the flow.
        flow_num_bins: Number of bins for the Rational Quadratic Spline.
    """
    def __init__(
        self,
        num_numerical_features: int,
        categorical_cardinalities: List[int],
        numerical_embedding_dim: int = 32,
        category_embedding_dim: int = 32,
        resnet_main_dim: int = 128,
        resnet_k_multiplier: int = 2, # Hidden dim = main_dim * k
        resnet_num_blocks: int = 3,
        resnet_dropout: float = 0.2,
        flow_num_transforms: int = 5,
        flow_hidden_features: int = 256,
        flow_num_bins: int = 8,
    ):
        super().__init__()
        self.num_numerical = num_numerical_features
        self.num_categorical = len(categorical_cardinalities)

        # 1. Numerical Feature Encoding (MLP per feature)
        self.numerical_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, numerical_embedding_dim),
                nn.ReLU(),
                nn.Dropout(resnet_dropout)
            ) for _ in range(num_numerical_features)
        ])

        # 2. Categorical Feature Embedding
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=card, embedding_dim=category_embedding_dim)
            for card in categorical_cardinalities
        ])

        # 3. ResNet Backbone
        total_embed_dim = (num_numerical_features * numerical_embedding_dim +
                          len(categorical_cardinalities) * category_embedding_dim)
        
        self.resnet = nn.Sequential(
            nn.Linear(total_embed_dim, resnet_main_dim),
            *[ResNetBlock(
                resnet_main_dim,
                resnet_k_multiplier,
                resnet_dropout
            ) for _ in range(resnet_num_blocks)]
        )

        # 4. Conditional Normalizing Flow Head (Rational Quadratic Spline)
        # Uses zuko library for a conditional RQS flow
        self.normalizing_flow = zuko.flows.NSF(
            features=1,                          # Univariate target TODO maybe check 2 for multivariate?
            context=resnet_main_dim,             # Conditioned on ResNet output
            transforms=flow_num_transforms,
            bins=flow_num_bins,
            hidden_features=[flow_hidden_features, flow_hidden_features]*2, # Hidden layers for spline params
            randperm=False, # Keep feature order for univariate case
        )

    def forward(
        self,
        x_num: torch.Tensor, # Shape: (batch_size, num_numerical_features)
        x_cat: torch.Tensor  # Shape: (batch_size, num_categorical_features), dtype=torch.long
    ):
        """
        Performs the forward pass to get the conditional flow and its context.

        Args:
            x_num: Tensor of numerical features. Assumes NaNs are filled (e.g., with mean/median).
            x_cat: Tensor of categorical features (as integer indices).

        Returns:
            The context tensor `z` (output of ResNet) needed for conditioning the flow.
              Shape: (batch_size, resnet_main_dim).
        """

        # Numerical features
        num_embeds = []
        for i in range(self.num_numerical):
            feat = x_num[:, i:i+1]  # (batch_size, 1)
            embed = self.numerical_encoders[i](feat)
            num_embeds.append(embed)

        # Categorical features
        cat_embeds = [
            emb(x_cat[:, i]) for i, emb in enumerate(self.categorical_embeddings)
        ]
        
        # Concatenate all embeddings
        combined = torch.cat(num_embeds + cat_embeds, dim=1)
        
        # ResNet processing
        z = self.resnet(combined)
        
        return z # Return context tensor

    def log_prob(self, y: torch.Tensor, x_num: torch.Tensor, x_cat: torch.Tensor):
        """Compute log probability of target y given inputs"""
        context = self.forward(x_num, x_cat)
        distribution = self.normalizing_flow(context)
        return distribution.log_prob(y)


    def sample(self, x_num: torch.Tensor, x_cat: torch.Tensor, num_samples: int = 100):
        """Draws samples from the learned conditional distribution p(y|context)."""
        context = self.forward(x_num, x_cat)
        distribution = self.normalizing_flow(context)
        return distribution.rsample((num_samples,))#.permute(1, 0, 2)
    
    def predict_mean_std(self, x_num: torch.Tensor, x_cat: torch.Tensor, num_mc_samples: int = 500):
        """
        Estimates the mean and standard deviation of the predictive distribution
        using Monte Carlo sampling.
        """
        samples = self.sample(x_num,x_cat,num_mc_samples)  # (num_samples, batch_size, 1)
        
        # Compute statistics
        mean = samples.mean(dim=0)        # (batch_size, 1)
        std = samples.std(dim=0)          # (batch_size, 1)
        return mean.squeeze(-1), std.squeeze(-1)  # (batch_size,)

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            device: torch.device,
            lr = 1e-4,
            weight_decay = 1e-5,
            num_epochs = 100,
            patience_early_stopping = 15,
            model_save_path = None, # "tabresflow_best.pth"
            dataset_key_for_save = None # For naming the saved model
           ):
        """
        Trains the TabResFlow model.
        """
        self.to(device)
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience_early_stopping//2, factor=0.5, verbose=False)


        best_val_loss = float('inf')
        best_model_state_dict = None
        epochs_no_improve = 0

        logger.info(f"Starting training on {device} for {num_epochs} epochs... LR={lr}, WD={weight_decay}")
        
        for epoch in range(num_epochs):
            self.train()
            total_train_loss = 0.0
            for batch_idx, (x_num_b, x_cat_b, y_b) in enumerate(train_loader):
                x_num_b, x_cat_b, y_b = x_num_b.to(device), x_cat_b.to(device), y_b.to(device)

                optimizer.zero_grad()
                log_probs = self.log_prob(y_b, x_num_b, x_cat_b)
                loss = -log_probs.mean()
                
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN or Inf loss in training: epoch {epoch+1}, batch {batch_idx}. Skipping.")
                    continue

                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else float('nan')

            # Validation
            avg_val_loss = evaluate_nll(
                model=self, 
                data_loader=val_loader, 
                device=device,
                current_epoch_num=epoch+1 
            )

            logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # scheduler.step(avg_val_loss)

            if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0

                if model_save_path and dataset_key_for_save:
                    save_name = model_save_path.replace(".pth", f"_{dataset_key_for_save}.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss,
                        # 'model_params': self.get_params()
                    }, save_name)
                    logger.info(f"New best val loss: {best_val_loss:.4f}. Model saved to {save_name}")

                else:
                    best_model_state_dict = copy.deepcopy(self.state_dict())
                    logger.info(f"*New best val loss: {best_val_loss:.4f}.")

            elif not np.isnan(avg_val_loss):
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience_early_stopping:
                logger.info(f"Early stopping at epoch {epoch+1} after {patience_early_stopping} epochs with no improvement.")
                break
        
        logger.info(f"Training finished. Best Validation NLL: {best_val_loss:.4f}")
        return best_val_loss, best_model_state_dict