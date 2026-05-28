import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import random

class ConceptVAE(nn.Module):
    def __init__(self, n_variables: int, n_concepts: int, 
                 concept_map: Dict[int, List[int]], beta: float, alpha_case: str = "uniform", 
                 use_bias: bool = True, hidden_size: int = 1000,
                 use_bias: bool = True, hidden_size: int = 1000, dropout_p: float = 0.1,
                 device: torch.device = torch.device("cpu")):
        super(ConceptVAE, self).__init__()
        
        self.n_variables = n_variables
        self.n_concepts = n_concepts
        self.beta = beta
        self.alpha_case = alpha_case

        # 1. Build the interpretable latent weights for initialization (K x M)
        # This implements the formula: z_i = (1-beta)*sum(x_j for j in g(i)) + beta*sum(alpha_k*sum(x_j for j in g(k)))
        latent_weights = self._build_latent_weights(concept_map, alpha_case)

        # Encoder: Shared hidden layers
        self.encoder = nn.Sequential(
            nn.Linear(n_variables, 1000, bias=use_bias),
            nn.Linear(n_variables, hidden_size, bias=use_bias),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=dropout_p)
        ).to(device)
        
        self.encoder_mu = nn.Linear(1000, n_concepts, bias=use_bias).to(device)
        self.encoder_logvar = nn.Linear(1000, n_concepts, bias=use_bias).to(device)
        self.encoder_mu = nn.Linear(hidden_size, n_concepts, bias=use_bias).to(device)
        self.encoder_logvar = nn.Linear(hidden_size, n_concepts, bias=use_bias).to(device)

        # Store the target linear mapping as a buffer for alignment evaluation and loss
        self.register_buffer('latent_weights', latent_weights.to(device))


        # Linear Decoder: standard trainable layer with no activation function
        self.decoder = nn.Linear(n_concepts, n_variables, bias=True).to(device)

        # 2. Build the decoder mask based on the concept mapping
        # This ensures that each concept neuron only influences its assigned variables
        decoder_mask = torch.zeros((n_variables, n_concepts))
        for c_idx, vars_indices in concept_map.items():
            for v_idx in vars_indices:
                decoder_mask[v_idx, c_idx] = 1.0
        self.register_buffer('decoder_mask', decoder_mask.to(device))

    def _build_latent_weights(self, concept_map: Dict[int, List[int]], alpha_case: str) -> torch.Tensor:
        K, M = self.n_concepts, self.n_variables
        weights = torch.zeros((K, M))
        V = torch.zeros((K, M))
        for c_idx, vars_indices in concept_map.items():
            V[c_idx, vars_indices] = 1.0

        for i in range(K):
            if alpha_case == "uniform":
                alphas = torch.full((K,), 1.0 / (K - 1))
            else:
                # Random exponential alphas for non-uniform case
                alphas_raw = torch.from_numpy(np.random.exponential(scale=1.0, size=K)).float()
                alphas_raw[i] = 0 
                alphas = alphas_raw / alphas_raw.sum()
            
            alphas[i] = 0 
            term1 = (1 - self.beta) * V[i]
            term2 = self.beta * (alphas.unsqueeze(0) @ V).squeeze(0)

            # z_i = (1-beta)*sum(x_j for j in g(i)) + beta*sum(alpha_k*sum(x_j for j in g(k)))
            # Removed normalization to strictly respect the evaluation criteria formula
            weights[i] = term1 + term2
        return weights

    def encode(self, x):
        # Pass through the shared encoder layers
        h = self.encoder(x)
        return self.encoder_mu(h), self.encoder_logvar(h)

    def to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Deterministic mapping of input data to latent embeddings (mean values).
        This avoids the stochasticity of the reparameterization trick.
        """
        mu, _ = self.encode(x)
        return mu

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Linear reconstruction (no activation function)
        # We apply the mask to the weights to enforce the concept-variable links
        masked_weight = self.decoder.weight * self.decoder_mask
        return nn.functional.linear(z, masked_weight, self.decoder.bias)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

def vae_loss(x, x_hat, mu, logvar, z_targets, 
             lambda_constraint=1.0, beta_kl=1.0):
    recon_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    constraint_loss = torch.nn.functional.mse_loss(mu, z_targets, reduction='mean')
    total = recon_loss + beta_kl * kl_loss + lambda_constraint * constraint_loss
    return total, recon_loss, kl_loss, constraint_loss

def compute_z_targets(x, latent_weights):
    # This function uses the pre-calculated matrix built in _build_latent_weights
    # which mathematically matches the snippet's local/global logic.
    return x @ latent_weights.t()

def train_and_validate(X_train, X_val, n_variables, n_concepts, concept_map, 
                       lr, kl_beta, lambda_align, beta_val, alpha_case, use_bias, dropout_p, 
                       num_epochs, batch_size, device, verbose=False):
    """
    Encapsulates the training and validation loop for a single HPO trial.
    """
    model = ConceptVAE(n_variables, n_concepts, concept_map, 
                       beta_val, alpha_case, use_bias=use_bias, 
                       beta_val, alpha_case, use_bias=use_bias, dropout_p=dropout_p,
                       device=device).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = TensorDataset(X_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    
    # Initialize history tracking
    train_history = {'total_loss': [], 'recon_loss': [], 'kld_loss': [], 'align_loss': []}
    val_history = {'total_loss': [], 'recon_loss': [], 'kld_loss': [], 'align_loss': []}

    for epoch in range(num_epochs):
        model.train()
        epoch_train_losses = np.zeros(4)
        batch_count = 0
        for batch in train_loader:
            inputs = batch[0].to(device)
            recon_batch, mu, logvar, z = model(inputs)
            
            # Calculate theoretical z values based on the formula for alignment
            z_targets = compute_z_targets(inputs, model.latent_weights)
            loss, r_loss, k_loss, a_loss = vae_loss(inputs, recon_batch, mu, logvar, z_targets, lambda_align, kl_beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_losses += np.array([loss.item(), r_loss.item(), k_loss.item(), a_loss.item()])
            batch_count += 1

        # Log training average loss
        for idx, key in enumerate(train_history.keys()):
            train_history[key].append(epoch_train_losses[idx] / batch_count)

        # Validation
        model.eval()
        with torch.no_grad():
            v_inputs = X_val.to(device)
            v_recon, v_mu, v_logvar, _ = model(v_inputs)
            v_target = compute_z_targets(v_inputs, model.latent_weights)
            v_loss, vr_loss, vk_loss, va_loss = vae_loss(v_inputs, v_recon, v_mu, v_logvar, v_target, lambda_align, kl_beta)
            val_loss = v_loss.item()
            
            # Log validation history
            val_history['total_loss'].append(v_loss.item())
            val_history['recon_loss'].append(vr_loss.item())
            val_history['kld_loss'].append(vk_loss.item())
            val_history['align_loss'].append(va_loss.item())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        if verbose and (epoch + 1) % 10 == 0: # Print more frequently for final training
            print(f"    Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.6f}")

    return best_val_loss, model, train_history, val_history

def optimize_hyperparameters(X_train, X_val, n_variables, n_concepts, concept_map, 
                             beta_val, alpha_case, use_bias, dropout_p, lambda_align, device, n_trials=10):
    """
    Performs a simple random search for learning rate and kl_beta.
    """
    print(f"\nStarting Hyperparameter Optimization ({n_trials} trials) for beta={beta_val}, lambda_align={lambda_align}...")
    
    # Define search space
    search_space = {
        'lr': [0.001, 0.005, 0.01],
        'kl_beta': [0.01, 0.05, 0.1, 0.2]
    }
    
    best_config = None
    min_val_loss = float('inf')
    best_model, best_train_history, best_val_history = None, None, None
    
    # We use fewer epochs for HPO to save time
    hpo_epochs = 60
    batch_size = 128

    # Using a deterministic subset of combinations if trials < search space size, 
    # or random sampling
    configs = []
    for _ in range(n_trials):
        configs.append({
    for i in range(n_trials):
        config = {
            'lr': random.choice(search_space['lr']),
            'kl_beta': random.choice(search_space['kl_beta'])
        })
        }

        print(f"Trial {i+1}/{n_trials}: lr={config['lr']}, kl_beta={config['kl_beta']}")
        
        val_loss, model, train_history, val_history = train_and_validate(
            X_train, X_val, n_variables, n_concepts, concept_map,
            lr=config['lr'], 
            kl_beta=config['kl_beta'],
            lambda_align=lambda_align,
            beta_val=beta_val,
            alpha_case=alpha_case,
            use_bias=use_bias,
            dropout_p=dropout_p,
            num_epochs=hpo_epochs,
            batch_size=batch_size,
            device=device
        )
        
        print(f"  Resulting Val Loss: {val_loss:.6f}")
        
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_config = config
            best_model = model
            best_train_history = train_history
            best_val_history = val_history
            
    print(f"\nOptimization Finished. Best Config: {best_config}, Best Val Loss: {min_val_loss:.6f}")
    return best_config, best_model, best_train_history, best_val_history

def plot_loss_curves(train_history, val_history, filename="loss_curves.png"):
    """
    Plots the training and validation loss curves for total, reconstruction, KLD, and alignment losses.
    """
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(train_history['total_loss']) + 1)

    plt.plot(epochs, train_history['total_loss'], label='Train Total Loss', color='blue')
    plt.plot(epochs, val_history['total_loss'], label='Val Total Loss', color='red')
    plt.plot(epochs, train_history['recon_loss'], label='Train Recon Loss', color='lightblue', linestyle='--')
    plt.plot(epochs, val_history['recon_loss'], label='Val Recon Loss', color='salmon', linestyle='--')
    plt.plot(epochs, train_history['kld_loss'], label='Train KLD Loss', color='green', linestyle=':')
    plt.plot(epochs, val_history['kld_loss'], label='Val KLD Loss', color='lightgreen', linestyle=':')
    
    # Only plot alignment loss if it was active (lambda_align > 0 and values are non-zero)
    if train_history['align_loss'] and any(l > 0 for l in train_history['align_loss']):
        plt.plot(epochs, train_history['align_loss'], label='Train Alignment Loss', color='purple', linestyle='-.')
        plt.plot(epochs, val_history['align_loss'], label='Val Alignment Loss', color='violet', linestyle='-.')

    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log') # Often losses are better viewed on a log scale
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Loss curves saved to: {filename}")
    plt.close()

def visualize_decoder_weights(model, filename="decoder_weights.png"):
    """
    Visualizes the learned decoder weights, highlighting the masked connections.
    """
    model.eval()
    with torch.no_grad():
        # Get the masked decoder weights
        masked_weight = (model.decoder_weight * model.mask.t()).t()
        masked_weight = model.decoder.weight * model.decoder_mask
        
        plt.figure(figsize=(10, 10))
        sns.heatmap(masked_weight.cpu().numpy(), cmap='viridis', cbar_kws={'label': 'Weight Value'})
        plt.title('Learned Decoder Weights (Masked)')
        plt.xlabel('Concept Index')
        plt.ylabel('Variable Index')
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Decoder weights visualization saved to: {filename}")
    plt.close()

def visualize_encoder_alignment(model, X_val, filename="encoder_alignment.png"):
    """
    Compares the learned latent space (mu) with the ideal linear mapping from input X.
    Plots Pearson correlation per concept and a heatmap comparison of effective encoder weights.
    """
    model.eval()
    with torch.no_grad():
        learned_mu = model.to_latent(X_val).cpu().numpy()
        ideal_mu_from_X = (X_val @ model.latent_weights.t()).cpu().numpy()

        correlations = np.zeros(model.n_concepts)
        for i in range(model.n_concepts):
            if np.std(learned_mu[:, i]) > 1e-6 and np.std(ideal_mu_from_X[:, i]) > 1e-6:
                correlations[i] = np.corrcoef(learned_mu[:, i], ideal_mu_from_X[:, i])[0, 1]
            else:
                correlations[i] = 0 # Handle cases with zero variance

        plt.figure(figsize=(12, 6))
        plt.bar(range(model.n_concepts), correlations, color='skyblue')
        plt.title('Pearson Correlation between Learned Latent Mu and Ideal Linear Mu per Concept')
        plt.xlabel('Concept Index')
        plt.ylabel('Correlation Coefficient')
        plt.ylim(-1, 1)
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Encoder alignment visualization (correlations) saved to: {filename}")
        plt.close()

        # Visualize effective encoder weights vs. ideal latent_weights
        effective_encoder_weights = (model.mu_layer.weight @ model.encoder_hidden.weight).cpu().numpy()
        effective_encoder_weights = (model.encoder_mu.weight @ model.encoder[0].weight).cpu().numpy()
        ideal_latent_weights = model.latent_weights.cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        sns.heatmap(ideal_latent_weights, cmap='coolwarm', center=0, ax=axes[0], cbar_kws={'label': 'Ideal Weight'})
        axes[0].set_title('Ideal Latent Weights (from formula)')
        axes[0].set_xlabel('Variable Index')
        axes[0].set_ylabel('Concept Index')

        sns.heatmap(effective_encoder_weights, cmap='coolwarm', center=0, ax=axes[1], cbar_kws={'label': 'Learned Effective Encoder Weights'})
        axes[1].set_xlabel('Variable Index')
        axes[1].set_ylabel('Concept Index')

        plt.tight_layout()
        plt.savefig("encoder_weights_comparison.png")
        print(f"Encoder weights comparison visualization saved to: encoder_weights_comparison.png")
        plt.close()


if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Assuming data_generation.py is in the same directory or accessible via PYTHONPATH
    from data_generation import generate_structured_dataset, save_concept_table_csv, visualize_results

    # Example Usage:
    # 1. Increase Examples: Higher N provides better signal for M=2000
    N_EXAMPLES, N_VARIABLES, N_CONCEPTS, OVERLAP_SKEW, MAX_SIZE = 10000, 2000, 100, 15.0, 100

    # Generate data
    X_np, concept_activities_np, concept_map = generate_structured_dataset(
        N_EXAMPLES, N_VARIABLES, N_CONCEPTS, overlap_skew=OVERLAP_SKEW, max_concept_size=MAX_SIZE
    )

    # 2. Input Standardization: Neural networks fail if inputs are not centered around 0.
    X_mean = X_np.mean(axis=0)
    X_std = X_np.std(axis=0)
    X_std[X_std == 0] = 1.0 
    X_scaled = (X_np - X_mean) / X_std

    print(f"Generated Dataset X shape: {X_np.shape}")
    print(f"Generated Concept Activities shape: {concept_activities_np.shape}")

    X_tensor = torch.from_numpy(X_scaled).float()

    # Split data for HPO and training (80% train, 20% validation)
    indices = list(range(N_EXAMPLES))
    random.shuffle(indices)
    split = int(0.8 * N_EXAMPLES)
    train_indices, val_indices = indices[:split], indices[split:]
    
    X_train = X_tensor[train_indices]
    X_val = X_tensor[val_indices]

    # Model Parameters
    beta_val = 0.1 
    alpha_case = "exponential" 
    use_bias = True
    dropout_p = 0.1
    lambda_align = 0.5  # Weight for the interpretable criteria alignment loss
    
    # 1. Run Hyperparameter Optimization
    best_params, best_model_hpo, best_train_history_hpo, best_val_history_hpo = optimize_hyperparameters(
        X_train, X_val, N_VARIABLES, N_CONCEPTS, concept_map, 
        beta_val, alpha_case, use_bias, dropout_p, lambda_align, device, n_trials=5
    )

    # 2. Final Training with best parameters
    print("\nStarting Final Reconstruction Training with Best Parameters...")
    num_epochs = 150
    batch_size = 128
    
    _, final_model, final_train_history, final_val_history = train_and_validate(
        X_train, X_val, N_VARIABLES, N_CONCEPTS, concept_map,
        lr=best_params['lr'],
        kl_beta=best_params['kl_beta'],
        lambda_align=lambda_align,
        beta_val=beta_val,
        alpha_case=alpha_case,
        use_bias=use_bias,
        dropout_p=dropout_p,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        verbose=True
    )

    print("Training Finished.")

    # 3. Plotting results
    print("\nGenerating visualizations...")
    plot_loss_curves(final_train_history, final_val_history, filename="final_loss_curves.png")
    visualize_decoder_weights(final_model, filename="final_decoder_weights.png")
    visualize_encoder_alignment(final_model, X_val, filename="final_encoder_alignment_correlations.png")

    final_model.eval()
    with torch.no_grad():
        inputs = X_tensor.to(device)
        recon_X, mu, _, _ = final_model(inputs)
        
        # 1. Reconstruction MSE
        # Since inputs are standardized (var=1), MSE is the proportion of unexplained variance.
        mse = nn.functional.mse_loss(recon_X, inputs).item()
        
        # 2. Pearson Correlation
        # We flatten the tensors to compute the global correlation between all input and output values.
        correlation = torch.corrcoef(torch.stack([inputs.flatten(), recon_X.flatten()]))[0, 1].item()
        
        print(f"\nFinal Evaluation Metrics:")
        print(f"Reconstruction MSE: {mse:.4f}")
        print(f"Pearson Correlation: {correlation:.4f}")
        print(f"Latent Mu shape: {mu.shape}")

    save_concept_table_csv(concept_map)
    visualize_results(X_np, concept_activities_np, concept_map, N_VARIABLES)
