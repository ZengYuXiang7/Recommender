# coding : utf-8
# Author : yuxiang Zeng
import torch


class MF(torch.nn.Module):
    def __init__(self, args):
        super(MF, self).__init__()
        self.args = args
        self.rank = args.rank
        self.dimension_gmf = args.rank
        self.embed_user_GMF = torch.nn.Embedding(162541 + 1, self.dimension_gmf)
        self.embed_item_GMF = torch.nn.Embedding(59047 + 1, self.dimension_gmf)

    def forward(self, userIdx, servIdx):
        user_embed = self.embed_user_GMF(userIdx)
        item_embed = self.embed_item_GMF(servIdx)
        prediction = torch.sum(user_embed * item_embed, dim=-1)
        return prediction
