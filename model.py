import torch.nn as nn
import transformers


class ViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, n_channels=3, n_classes=10):
        super().__init__()
        vit_args = transformers.ViTConfig(hidden_size=256, num_hidden_layers=6, num_attention_heads=4,
                                          intermediate_size=256 * 2, hidden_act='gelu', hidden_dropout_prob=0.1,
                                          attention_probs_dropout_prob=0.0, initializer_range=0.02,
                                          layer_norm_eps=1e-12, is_encoder_decoder=False, image_size=image_size,
                                          patch_size=patch_size, num_channels=n_channels, qkv_bias=True)

        self.encoder = transformers.ViTModel(vit_args)
        self.clf = nn.Linear(256, n_classes)

    def forward(self, x, deep=False):
        x = self.encoder(x)
        x_deep = x[1]
        x = self.clf(x_deep)
        if deep:
            return x, x_deep
        return x
