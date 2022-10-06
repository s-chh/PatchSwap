import torch.nn as nn
import transformers


class ViT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        vit_args = transformers.ViTConfig(hidden_size=256, num_hidden_layers=6, num_attention_heads=4,
                                          intermediate_size=256 * 2, hidden_act='gelu', hidden_dropout_prob=0.1,
                                          attention_probs_dropout_prob=0.0, initializer_range=0.02,
                                          layer_norm_eps=1e-12, is_encoder_decoder=False, image_size=args.img_size,
                                          patch_size=args.patch_size, num_channels=args.num_channels, qkv_bias=True)

        self.encoder = transformers.ViTModel(vit_args)
        self.clf = nn.Linear(256, args.num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = x[1]
        x = self.clf(x)
        return x
