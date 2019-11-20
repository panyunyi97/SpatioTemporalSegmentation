import torch
from torch import nn
import sys
from MinkowskiEngine.SparseTensor import SparseTensor
sa_module_on = True
sc_module_on = True


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim, batch_size=1):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.batch_size = batch_size

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize = 1
        fx = x.feats
        x_key=x.coords_key
        x_cm=x.coords_man
        x_stride=x.tensor_stride
        N, C = fx.shape
        #locations = x.metadata.getSpatialLocations(x.spatial_size)
        locations=x.coords
        batch_size_list = []
        final_list = []
        
        for i in range(0, self.batch_size):
            batch_size_list.append((locations[:, 3] == i).nonzero().shape[0])
        total_count = 0

        for batch_size in batch_size_list:
            if batch_size == 0:
                break
            single_x = torch.index_select(fx, 0, torch.arange(
                total_count, total_count + batch_size).cuda())
            total_count += batch_size
            proj_query = self.query_conv(
                single_x.unsqueeze(-1)).view(m_batchsize, -1, batch_size).permute(0, 2, 1)
            proj_key = self.key_conv(
                single_x.unsqueeze(-1)).view(m_batchsize, -1, batch_size)
            energy = torch.bmm(proj_query, proj_key)
            attention = self.softmax(energy)
            proj_value = self.value_conv(
                single_x.unsqueeze(-1)).view(m_batchsize, -1, batch_size)
            out = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out = out.view(batch_size, C)
            out = self.gamma * out + single_x
            final_list.append(out)
        final_list = torch.cat(final_list, 0)
        x=SparseTensor(final_list,coords=locations,coords_key=x_key,coords_manager=x_cm,tensor_stride=x_stride) 
        return x


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim, batch_size=1):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.batch_size = batch_size

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize = 1
        fx = x.feats
        x_key=x.coords_key
        x_cm=x.coords_man
        x_stride=x.tensor_stride
        N, C = fx.shape
        #locations = x.metadata.getSpatialLocations(x.spatial_size)
        locations=x.coords
        batch_size_list = []
        final_list = []
        for i in range(0, self.batch_size):
            batch_size_list.append((locations[:, 3] == i).nonzero().shape[0])
        total_count = 0
        for batch_size in batch_size_list:
            if batch_size == 0:
                break
            single_x = torch.index_select(fx, 0, torch.arange(
                total_count, total_count + batch_size).cuda())
            total_count += batch_size

            proj_query = single_x.unsqueeze(0).view(m_batchsize, C, -1)
            proj_key = single_x.unsqueeze(0).view(
                m_batchsize, C, -1).permute(0, 2, 1)
            energy = torch.bmm(proj_query, proj_key)
            energy_new = torch.max(
                energy, -1, keepdim=True)[0].expand_as(energy) - energy
            attention = self.softmax(energy_new)
            proj_value = single_x.unsqueeze(0).view(m_batchsize, C, -1)

            out = torch.bmm(attention, proj_value)
            out = out.view(batch_size, C)
            out = self.gamma * out + single_x
            final_list.append(out)
        final_list = torch.cat(final_list, 0)
        x=SparseTensor(final_list,coords=locations,coords_key=x_key,coords_manager=x_cm,tensor_stride=x_stride) 
        return x


class DANetHead(nn.Module):
    def __init__(self, in_channels=112, out_channels=112, norm_layer=nn.BatchNorm1d, batch_size=1):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4

        self.sa = PAM_Module(in_channels, batch_size)
        self.sc = CAM_Module(in_channels, batch_size)

    def forward(self, x):

        if sa_module_on:
            sa_conv = self.sa(x)
        if sc_module_on:
            sc_conv = self.sc(x)
        if not sa_module_on and not sc_module_on:
            pass
        else:
            features = (sa_conv.feats if sa_module_on else 0) + \
                         (sc_conv.feats if sc_module_on else 0)
        
        x_key=sa_conv.coords_key
        x_cm=sa_conv.coords_man
        x_stride=sa_conv.tensor_stride
        locations=sa_conv.coords
        x=SparseTensor(features,coords=locations,coords_key=x_key,coords_manager=x_cm,tensor_stride=x_stride) 
        return x
