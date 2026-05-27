import torch
import torch.nn as nn

class FeatureConstrainedAutoencoder(nn.Module):
    def __init__(self, seq_len=101, in_channels=2, latent_dim=8, num_features=11):
        super(FeatureConstrainedAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        # --- Encoder ---
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.flatten_dim = 64 * 13
        self.encoder_linear = nn.Sequential(
            nn.Linear(self.flatten_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # --- Decoder ---
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.flatten_dim),
            nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=in_channels, kernel_size=5, stride=2, padding=2, output_padding=0),
        )

        # --- Feature Predictor Branch ---
        self.feature_predictor = nn.Linear(latent_dim, num_features)
        
    def forward(self, x):
        # x shape: (Batch, seq_len, in_channels)
        x = x.transpose(1, 2) # (Batch, 2, 101)
        
        # Encode
        x_enc = self.encoder_conv(x)
        x_enc = x_enc.view(x_enc.size(0), -1)
        z = self.encoder_linear(x_enc)
        
        # Decode
        x_dec = self.decoder_linear(z)
        x_dec = x_dec.view(x_dec.size(0), 64, 13)
        x_recon = self.decoder_conv(x_dec)
        x_recon = x_recon.transpose(1, 2) # (Batch, 101, 2)
        
        # Predict Features
        pred_features = self.feature_predictor(z)
        
        return x_recon, z, pred_features
