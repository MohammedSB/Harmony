import torch
import torch.nn as nn
from Harmony.models.transformer import Transformer, LayerNorm

class TextEncoder(torch.nn.Module):
    def __init__(self, context_length=77, vocab_size=49409, transformer_width=512, transformer_heads=8,
                 transformer_layers=12, embed_dim=512):
        super(TextEncoder, self).__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads 
        self.transformer_layers = transformer_layers
        self.embed_dim = embed_dim
        
        self.backbone = Transformer(
            width=self.transformer_width,
            layers=self.transformer_layers,
            heads=self.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.context_length = self.context_length + 1

        self.text_embedding = torch.nn.Embedding(self.vocab_size, self.transformer_width)
        self.text_positional_embedding = torch.nn.Parameter(torch.empty(self.context_length, self.transformer_width))
        self.text_norm = LayerNorm(self.transformer_width)
        self.text_projection = torch.nn.Parameter(torch.empty(self.transformer_width, self.embed_dim), requires_grad=True)

        self.init()

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def init(self):
        nn.init.normal_(self.text_embedding.weight, std=0.02)
        # nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.text_positional_embedding, std=0.01)

        proj_std = (self.backbone.width ** -0.5) * ((2 * self.backbone.layers) ** -0.5)
        attn_std = self.backbone.width ** -0.5
        fc_std = (2 * self.backbone.width) ** -0.5
        for block in self.backbone.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.text_projection, std=self.backbone.width ** -0.5)

    def forward(self, text, return_without_proj=False):
        x = self.text_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.text_positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.backbone(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        o = self.text_norm(x)
        o = o[torch.arange(o.shape[0]), text.argmax(dim=-1)] 
        
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)

        o = o @ self.text_projection
        if return_without_proj:
            return o, x
        return o
