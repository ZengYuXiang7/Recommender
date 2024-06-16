# coding : utf-8
# Author : yuxiang Zeng
import torch

class NeuCF(torch.nn.Module):
    def __init__(self, args):
        super(NeuCF, self).__init__()
        self.args = args
        self.num_layers = args.num_layers
        self.dropout = 0.10
        self.dimension = args.dimension
        self.dimension_gmf = args.dimension
        self.dimension_mlp = args.dimension * (2 ** (self.num_layers - 1))
        self.embed_user_GMF = torch.nn.Embedding(162541 + 1, self.dimension_gmf)
        self.embed_user_MLP = torch.nn.Embedding(162541 + 1, self.dimension_mlp)
        self.embed_item_GMF = torch.nn.Embedding(59047 + 1, self.dimension_gmf)
        self.embed_item_MLP = torch.nn.Embedding(59047 + 1, self.dimension_mlp)

        MLP_modules = []
        for i in range(self.num_layers):
            input_size = self.dimension * (2 ** (self.num_layers - i))
            MLP_modules.append(torch.nn.Dropout(p=self.dropout))
            MLP_modules.append(torch.nn.Linear(input_size, input_size // 2))
            MLP_modules.append(torch.nn.ReLU())
        self.MLP_layers = torch.nn.Sequential(*MLP_modules)
        self.predict_layer = torch.nn.Linear(self.dimension * 2, 1)


    def forward(self, userIdx, servIdx):
        user_embed = self.embed_user_GMF(userIdx)
        embed_user_MLP = self.embed_user_MLP(userIdx)

        item_embed = self.embed_item_GMF(servIdx)
        embed_item_MLP = self.embed_item_MLP(servIdx)

        gmf_output = user_embed * item_embed
        mlp_input = torch.cat((embed_user_MLP, embed_item_MLP), -1)

        mlp_output = self.MLP_layers(mlp_input)

        prediction = self.predict_layer(torch.cat((gmf_output, mlp_output), -1))

        return prediction.flatten()