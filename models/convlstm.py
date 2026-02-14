import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        conv_out = self.conv(combined)

        i, f, o, g = torch.chunk(conv_out, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: input = optical flow (3 channels)
        self.encoder = ConvLSTMCell(input_channels=3, hidden_channels=64)

        # Decoder: input = encoded feature maps (64 channels)
        self.decoder = ConvLSTMCell(input_channels=64, hidden_channels=64)

        # Final reconstruction layer
        self.output_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        device = x.device

        h = torch.zeros(B, 64, H, W, device=device)
        c = torch.zeros(B, 64, H, W, device=device)

        encoded_states = []

        # ---- Encoder ----
        for t in range(T):
            h, c = self.encoder(x[:, t], h, c)
            encoded_states.append(h)

        # ---- Decoder ----
        decoded = []
        h_dec, c_dec = h, c

        for t in range(T):
            h_dec, c_dec = self.decoder(h_dec, h_dec, c_dec)
            out = self.output_conv(h_dec)
            decoded.append(out)

        decoded = torch.stack(decoded, dim=1)
        return decoded
