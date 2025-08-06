from .grassmann import *
from args_fusion import args

class signal2spd(nn.Module):
    # convert signal epoch to SPD matrix
    def __init__(self):
        super().__init__()
        self.dev = torch.device('cpu')

    def forward(self, x):

        # x = x.squeeze()

        # mean = x.mean(axis=-1).unsqueeze(-1).repeat(1, 1, x.shape[-1])
        # x = x - mean
        cov = x @ x.transpose(-1, -2)
        cov = cov.to(self.dev)
        cov = cov / (x.shape[-1] - 1)

        tra = cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        # tra = tra.unsqueeze(0).unsqueeze(3)

        tra = tra.unsqueeze(2).unsqueeze(3)

        # tra = tra.view(-1, 1, 1)

        cov /= tra

        # print(cov.shape)
        # # identity = torch.eye(cov.shape[-1], cov.shape[-1], device=self.dev).repeat(x.shape[0], 1, 1)
        # identity = torch.eye(cov.shape[-1], cov.shape[-1], device=self.dev).repeat(x.shape[0], 1, 1)
        # identity2 = torch.eye(cov.shape[-1], cov.shape[-1], device=self.dev).repeat(x.shape[0],1,1,1)
        # print(identity2.shape)
        # cov = cov + (1e-5 * identity)
        return cov

class signal2spd2(nn.Module):
    # convert signal epoch to SPD matrix
    def __init__(self):
        super().__init__()
        self.dev = torch.device('cpu')

    def forward(self, x):
        # x = x.squeeze()
        # mean = x.mean(axis=-1).unsqueeze(-1).repeat(1, 1, x.shape[-1])
        # x = x - mean
        cov = x @ x.transpose(-1, -2)
        cov = cov.to(self.dev)
        cov = cov / (x.shape[-1] - 1)
        tra = cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

        tra = tra.unsqueeze(3).unsqueeze(4)

        # tra = tra.view(8,8, 1, 1)
        cov /= tra

        # identity = torch.eye(cov.shape[-1], cov.shape[-1], device=self.dev).repeat(x.shape[0], 1, 1)
        # cov = cov + (1e-5 * identity)
        return cov

# class signal2spd2(nn.Module):
#     # convert signal epoch to SPD matrix
#     def __init__(self):
#         super().__init__()
#         self.dev = torch.device('cpu')
#
#     def forward(self, x):
#         print(x.shape)
#         x = x.squeeze()
#         # mean = x.mean(axis=-1).unsqueeze(-1).repeat(1, 1, x.shape[-1])
#         # x = x - mean
#         cov = x @ x.transpose(-1, -2)
#         cov = cov.to(self.dev)
#         cov = cov / (x.shape[-1] - 1)
#
#         tra = cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
#         tra = tra.view(8,8, 1, 1)
#         cov /= tra
#
#         identity = torch.eye(cov.shape[-1], cov.shape[-1], device=self.dev).repeat(x.shape[0], 1, 1)
#         cov = cov + (1e-5 * identity)
#         return cov


class E2R(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.proj = signal2spd()

    @staticmethod
    def patch_len(n, epochs):
        list_len = []
        base = n // epochs
        for i in range(epochs):
            list_len.append(base)
        for i in range(n - base * epochs):
            list_len[i] += 1
        if sum(list_len) == n:
            return list_len
        else:
            return ValueError('check your epochs and axis should be split again')

    def forward(self, x):

        # x = x.squeeze()
        list_patch = self.patch_len(x.shape[-2], int(self.epochs))
        x_list = list(torch.split(x, list_patch, dim=-2))
        for i, item in enumerate(x_list):
            x_list[i] = self.proj(item)
        x = torch.stack(x_list)

        if x.ndim == 5:
            x = x.squeeze(0)

        if x.ndim == 3:
            x = x.unsqueeze(1)

        # x = x.permute(1, 0, 2, 3)
        return x

class E2R2(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.proj = signal2spd2()

    @staticmethod
    def patch_len(n, epochs):
        list_len = []
        base = n // epochs
        for i in range(epochs):
            list_len.append(base)
        for i in range(n - base * epochs):
            list_len[i] += 1
        if sum(list_len) == n:
            return list_len
        else:
            return ValueError('check your epochs and axis should be split again')

    def forward(self, x):
        # x = x.squeeze()
        list_patch = self.patch_len(x.shape[-2], int(self.epochs))
        x_list = list(torch.split(x, list_patch, dim=-2))

        for i, item in enumerate(x_list):
            x_list[i] = self.proj(item)




        x = torch.stack(x_list)
        if x.ndim == 6:
            x = x.squeeze(0)


        # print(x.shape)
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # x = x.permute(1, 0, 2, 3)
        return x


class AttentionManifold(nn.Module):
    def __init__(self, in_size, out_size):
        super(AttentionManifold, self).__init__()
        self.q_trans = FRMap(in_size, out_size).cpu()
        self.k_trans = FRMap(in_size, out_size).cpu()
        self.v_trans = FRMap(in_size, out_size).cpu()
        self.qr = QRComposition()
        self.proj = Projmap()

    @staticmethod
    def projection_metric(x, y):
        """Compute the projection metric between matrices A and B"""

        inner_term = torch.matmul(x, x.transpose(-1, -2)) - torch.matmul(y, y.transpose(-1, -2))

        # print(torch.matmul(x, x.transpose(-1, -2)).shape)
        return torch.norm(inner_term, dim=-1)

    @staticmethod
    def w_frechet_distance_mean(W, V):
        """Compute the weighted Fréchet mean based on PM (projection metric)"""

        W = W.double().unsqueeze(-1).unsqueeze(-1)
        V = V.unsqueeze(1)
        res = V * W
        return res.sum(dim=2)

    def forward(self, x):

        Q = self.qr(self.q_trans(x).permute(0, 2,1))
        K = self.qr(self.k_trans(x).permute(0, 2,1))
        V = self.qr(self.v_trans(x).permute(0, 2,1))

        Q = Q.permute(0,2,1)
        K = K.permute(0, 2, 1)
        V = V.permute(0, 2, 1)
        # print(Q.shape)
        atten_energy = self.projection_metric(Q.unsqueeze(1), K.unsqueeze(2))

        atten_prob = nn.Softmax(dim=-2)(1 / (1 + torch.log(1 + atten_energy))).transpose(-1, -2)
        # print(atten_energy.shape)
        # print(V.shape)
        # V = self.proj(V)

        # output = self.w_frechet_distance_mean(atten_prob, V)
        output = atten_prob @ V

        return output


class GmAtt_mamem(nn.Module):
    def __init__(self, p):
        super().__init__()
        # Feature Extraction Module
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 125, (8, 1)), nn.BatchNorm2d(125), nn.ELU())
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(125, 21, (1, 64)), nn.BatchNorm2d(21), nn.ELU())
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(125, 21, (1, 64)), nn.BatchNorm2d(21), nn.ELU())
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(125, 21, (1, 64)), nn.BatchNorm2d(21), nn.ELU())

        # Manifold Modeling Module
        self.E2R = E2R(3)
        self.orth1 = Orthmap(p)

        # Grassmann Manifold Attention Module
        self.att1 = AttentionManifold(21, 19)
        # FC
        self.flat = nn.Flatten()
        self.linear = nn.Linear(3 * 19 * 19, 5, bias=True, dtype=torch.double)  # FC

    def forward(self, x):
        x = self.conv_block1(x)
        x1 = self.conv_block2(x)
        x2 = self.conv_block3(x)
        x3 = self.conv_block4(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.E2R(x.double())
        x = self.orth1(x)
        x = self.att1(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.flat(x)
        x = self.linear(x)
        return x


class GmAtt_cha(nn.Module):
    def __init__(self, p):
        super().__init__()
        # Feature Extraction Module
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 14, (56, 1)), nn.BatchNorm2d(14), nn.ELU())
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(14, 16, (1, 64)), nn.BatchNorm2d(16), nn.ELU())
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(14, 16, (1, 64)), nn.BatchNorm2d(16), nn.ELU())
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(14, 16, (1, 64)), nn.BatchNorm2d(16), nn.ELU())

        # Manifold Modeling Module
        self.E2R = E2R(3)
        self.orth = Orthmap(p)

        # Grassmann Manifold Attention Module
        self.att1 = AttentionManifold(16, 14)

        # FC
        self.flat = nn.Flatten()
        self.linear = nn.Linear(3 * 14 * 14, 2, bias=True, dtype=torch.double)

    def forward(self, x):
        x = self.conv_block1(x)
        x1 = self.conv_block2(x)
        x2 = self.conv_block3(x)
        x3 = self.conv_block4(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.E2R(x.double())
        x = self.orth(x)
        x = self.att1(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.flat(x)
        x = self.linear(x)
        return x
