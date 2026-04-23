import torch
import torch.nn as nn

class TrajectoryAutoencoder(nn.Module):
    def __init__(self, seq_len=101, in_channels=2, latent_dim=8):
        super(TrajectoryAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        # --- Encoder ---
        # Input shape: (Batch, Channels=2, Length=101)
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            # output length: ceil(101/2) = 51
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            # output length: ceil(51/2) = 26
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # output length: ceil(26/2) = 13
        )
        
        self.flatten_dim = 64 * 13
        
        self.encoder_linear = nn.Sequential(
            nn.Linear(self.flatten_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # --- Decoder ---
        # We need to project latent_dim back to flatten_dim
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.flatten_dim),
            nn.ReLU()
        )
        
        # Inverse of encoder convolutions. We use output_padding to get exact original length.
        # Length sequence backwards: 13 -> 26 -> 51 -> 101
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            # L_out = (13 - 1)*2 - 2*1 + 3 + 1 = 26
            nn.ReLU(),
            
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=0),
            # L_out = (26 - 1)*2 - 2*2 + 5 + 0 = 51
            nn.ReLU(),
            
            nn.ConvTranspose1d(in_channels=16, out_channels=in_channels, kernel_size=5, stride=2, padding=2, output_padding=0),
            # L_out = (51 - 1)*2 - 2*2 + 5 + 0 = 101
        )
        
    def forward(self, x):
        # x shape: (Batch, seq_len, in_channels)
        # PyTorch Conv1d expects: (Batch, in_channels, seq_len)
        x = x.transpose(1, 2)
        
        # Encode
        z = self.encode(x)
        
        # Decode
        x_recon = self.decode(z)
        
        # Transpose back: (Batch, seq_len, in_channels)
        x_recon = x_recon.transpose(1, 2)
        return x_recon, z
        
    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        z = self.encoder_linear(x)
        return z
        
    def decode(self, z):
        x = self.decoder_linear(z)
        x = x.view(x.size(0), 64, 13)
        x = self.decoder_conv(x)
        return x

if __name__ == "__main__":
    # Quick Test
    model = TrajectoryAutoencoder()
    dummy_input = torch.randn(5, 101, 2) # Batch=5, Length=101, Channels=2
    recon, latent = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Latent shape:", latent.shape)
    print("Reconstructed shape:", recon.shape)
