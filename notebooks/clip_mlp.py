import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
import os

# Add the parent directory to the path to import bayesvlm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayesvlm.utils import load_model, get_model_type_and_size
from bayesvlm.vlm import CLIPImageEncoder, CLIPTextEncoder


class MLPHead(nn.Module):
    """MLP head with dropout for probabilistic inference"""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        dropout_rate: float = 0.1,
        num_layers: int = 2
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        layers = []
        current_dim = input_dim
        
        # Create hidden layers
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        # Final layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
    
    def forward_with_dropout(self, x: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        """Forward pass with multiple dropout samples for uncertainty estimation"""
        self.train()  # Enable dropout
        outputs = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.forward(x)
                outputs.append(output)
        
        # Stack outputs: [n_samples, batch_size, num_classes]
        return torch.stack(outputs, dim=0)


class CLIPWithMLP(nn.Module):
    """CLIP model with trainable MLP head for classification"""
    
    def __init__(
        self, 
        clip_model_name: str = "clip-base",  # Use bayesvlm naming convention
        num_classes: int = 1000,
        mlp_hidden_dim: int = 512,
        mlp_dropout_rate: float = 0.1,
        mlp_num_layers: int = 2,
        freeze_clip: bool = True,
        device: str = 'cuda'
    ):
        super().__init__()
        
        # Load pretrained CLIP model using bayesvlm
        self.image_encoder, self.text_encoder, self.clip_model = load_model(
            clip_model_name, device=device
        )
        
        # Freeze CLIP parameters if specified
        if freeze_clip:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Get embedding dimension from CLIP
        # For CLIP models, the embedding dimension is typically 512 for base models
        model_type, model_size = get_model_type_and_size(clip_model_name)
        if model_size == "base":
            embed_dim = 512
        elif model_size == "large":
            embed_dim = 768
        elif model_size == "huge":
            embed_dim = 1024
        else:
            embed_dim = 512  # default
        
        # Add MLP head for classification
        self.mlp_head = MLPHead(
            input_dim=embed_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=num_classes,
            dropout_rate=mlp_dropout_rate,
            num_layers=mlp_num_layers
        )
        
        self.num_classes = num_classes
        self.device = device
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode images using CLIP visual encoder"""
        # Convert tensor to the expected batch format for bayesvlm
        batch = {'image': image}
        return self.image_encoder(batch)
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text using CLIP text encoder"""
        return self.text_encoder(text)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass through CLIP + MLP"""
        # Get image features from CLIP
        image_features = self.encode_image(image)
        
        # Pass through MLP head
        logits = self.mlp_head(image_features)
        
        return logits
    
    def forward_with_uncertainty(
        self, 
        image: torch.Tensor, 
        n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation using dropout
        
        Returns:
            mean_logits: Mean predictions across dropout samples
            std_logits: Standard deviation across dropout samples
        """
        # Get image features from CLIP (deterministic)
        with torch.no_grad():
            image_features = self.encode_image(image)
        
        # Get multiple predictions with dropout
        logits_samples = self.mlp_head.forward_with_dropout(image_features, n_samples)
        
        # Compute statistics
        mean_logits = logits_samples.mean(dim=0)
        std_logits = logits_samples.std(dim=0)
        
        return mean_logits, std_logits
    
    def predict_with_uncertainty(
        self, 
        image: torch.Tensor, 
        n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get predictions with uncertainty measures
        
        Returns:
            mean_probs: Mean probabilities
            epistemic_uncertainty: Epistemic uncertainty (predictive entropy)
            aleatoric_uncertainty: Aleatoric uncertainty (mutual information)
        """
        mean_logits, std_logits = self.forward_with_uncertainty(image, n_samples)
        
        # Convert to probabilities
        mean_probs = F.softmax(mean_logits, dim=-1)
        
        # Compute epistemic uncertainty (predictive entropy)
        epistemic_uncertainty = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        
        # Compute aleatoric uncertainty (mutual information approximation)
        # This is a simplified approximation using the variance in logits
        aleatoric_uncertainty = torch.mean(std_logits, dim=-1)
        
        return mean_probs, epistemic_uncertainty, aleatoric_uncertainty