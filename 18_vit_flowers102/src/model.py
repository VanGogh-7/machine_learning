import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 3,
        embedding_dim: int = 192,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        # A strided convolution extracts non-overlapping patches and projects
        # each patch into the Transformer embedding dimension.
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.projection(images)
        # [B, E, H/P, W/P] -> [B, num_patches, E]
        return patches.flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 102,
        embedding_dim: int = 192,
        num_heads: int = 6,
        mlp_dim: int = 384,
        num_encoder_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dim=embedding_dim,
        )
        num_patches = self.patch_embedding.num_patches

        # The class token is a learnable summary token used for classification.
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        # Positional embeddings tell the Transformer where each image patch is.
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embedding_dim)
        )
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        # The Transformer Encoder models relationships between all patch tokens.
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patch_embeddings = self.patch_embedding(images)
        batch_size = patch_embeddings.size(0)

        class_tokens = self.class_token.expand(batch_size, -1, -1)
        # CNNs process local neighborhoods; ViT treats image patches as tokens.
        tokens = torch.cat([class_tokens, patch_embeddings], dim=1)
        tokens = tokens + self.position_embedding
        tokens = self.dropout(tokens)

        encoded_tokens = self.encoder(tokens)
        encoded_tokens = self.norm(encoded_tokens)
        class_token_output = encoded_tokens[:, 0]
        return self.head(class_token_output)


if __name__ == "__main__":
    x = torch.randn(4, 3, 128, 128)
    model = VisionTransformer(
        image_size=128,
        patch_size=16,
        num_classes=102,
        embedding_dim=192,
        num_heads=6,
        mlp_dim=384,
        num_encoder_layers=6,
    )
    y = model(x)
    print(y.shape)
