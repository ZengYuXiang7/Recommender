# coding : utf-8
# Author : yuxiang Zeng
import torch


class CF(torch.nn.Module):
    def __init__(self, args):
        super(CF, self).__init__()
        self.args = args
        self.rank = args.rank
        self.user_embedding = torch.nn.Embedding(162541 + 1, self.dimension_gmf)
        self.item_embedding = torch.nn.Embedding(59047 + 1, self.dimension_gmf)

    def forward(self, userIdx, servIdx):
        user_embeds = self.user_embedding(userIdx)
        item_embeds = self.item_embedding(servIdx)
        prediction = torch.sum(user_embeds * item_embeds, dim=-1)
        return prediction
