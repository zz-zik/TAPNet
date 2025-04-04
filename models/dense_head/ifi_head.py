"""
https://github.com/AaronCIH/APGCC/blob/main/apgcc/models/Decoder.py
https://arxiv.org/pdf/2405.10589
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# IFI decoder
class IFI_Decoder_Model(nn.Module):
    def __init__(self, feat_layers=[3, 4], num_classes=2, num_anchor_points=4, line=2, row=2,
                 anchor_stride=None, inner_planes=256,
                 sync_bn=False, require_grad=False, head_layers=[512, 256, 256], out_type='Normal',
                 pos_dim=32, ultra_pe=False, learn_pe=False, unfold=False, local=False, stride=1,
                 **kwargs):
        """
        in_planes: encoder_outplanes; feat_layers: use_encoder_feats; inner_planes: trans_ens_outplanes;
        dilations: only for ASPP; out_type: type of output layer.
        num_anchor_points = line*row
        num_classes = num of classes

        -- For IfI modules:
        # feat_num: # of encoder feats; num_anchor_points: line*row, num of queries each pixel;
        # num_classes: 2 {'regression':2(momentum), 'classifier':confidence},
        # ultra_pe/learn_pe: additional position encoding, pos_dim: additional position encoding dimension
        # local/unfold: impose the local feature information, head_layers: head layers setting, defaults=[512,256,256]
        # feat_dim: input_feats_dims=inner_planes

        -- forward: {'pred_logits', 'pred_points', 'offset'}
        """
        super(IFI_Decoder_Model, self).__init__()
        self.inner_planes = inner_planes  # default: 256
        self.num_anchor_points = num_anchor_points
        self.num_classes = num_classes  # default: 2

        self.feat_num = len(feat_layers)  # change the encoder feature num.
        self.feat_layers = feat_layers  # control the number of decoder features.
        self.unfold = unfold
        self.out_type = out_type
        self.num_classes = num_classes

        # IFI Module: position_encoding + regression + classifier
        self.ifi = ifi_simfpn(ultra_pe=ultra_pe, pos_dim=pos_dim, sync_bn=sync_bn,
                              num_anchor_points=self.num_anchor_points, num_classes=self.num_classes,
                              local=local, unfold=unfold, stride=stride, learn_pe=learn_pe,
                              require_grad=require_grad, head_layers=head_layers, feat_num=self.feat_num,
                              feat_dim=inner_planes)
        # Output Neck.
        if self.out_type == 'Conv':
            raise NotImplemented
        elif self.out_type == "Deconv":
            raise NotImplemented

        # Align to real coords
        self.anchor_stride, self.row, self.line = anchor_stride, row, line
        # TODO:self.feat_layers[0]为只有w/16的层,而self.feat_layers[1]为可以为w/32的层
        self.anchor_points = AnchorPoints(pyramid_levels=self.feat_layers[1], stride=anchor_stride, row=row, line=line)

        # Auxiliary Anchors
        self.aux_en = kwargs['AUX_EN']
        self.aux_number = kwargs['AUX_NUMBER']
        self.aux_range = kwargs['AUX_RANGE']
        self.aux_kwargs = kwargs['AUX_kwargs']
        print("auxiliary anchors: (pos, neg, lambda, kwargs) ", self.aux_number, self.aux_range, self.aux_kwargs)

    def forward(self, samples, target_feat: list):
        # feats is [feat1, feat2, feat3, feat4]
        # align to max_shape of input_feat by implicit function
        ht, wt = target_feat[0].shape[-2], target_feat[0].shape[-1]  # 获取输入图像的尺寸
        # print("ht, wt: ", ht, wt)
        batch_size = target_feat[0].shape[0]

        # IFI module forward: position_encoding + offset_head + confidence_head
        context = []
        for i, feat in enumerate(target_feat):
            # print(feat.shape)
            # print(self.ifi(feat, size=[ht, wt], level=i + 1).shape)
            context.append(
                self.ifi(feat, size=[ht, wt], level=i + 1))  # context shape: [batch_size, h/d*w/d*line*row, 2(x,y)]
        # print(context[0].shape, context[1].shape, context[2].shape)
        context = torch.cat(context, dim=-1).permute(0, 2, 1)  # [b, h/d*w/d*line*row, 2(x,y)]
        offset, confidence = self.ifi(context, size=[ht, wt], after_cat=True)

        # output decoder.
        if self.out_type == 'Conv':
            raise KeyError('{} is not finished'.format(self.out_type))
        elif self.out_type == 'Deconv':
            raise KeyError('{} is not finished'.format(self.out_type))

        # Transform output
        offset *= 100
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)  # get sample point map
        output_coord = offset + anchor_points  # [b, h/d*w/d*line*row, 2(x,y)]  # transfer feature coordinate moving to image coordinate. (correspond to head center)
        output_confid = confidence  # [b, h/d*w/d*line*row, 2(confidence)]
        # out[pred_logits].shape: [b, h/d*w/d*line*row, 2(confidence)], out[pred_points].shape: [b, h/d*w/d*line*row, 2(x,y)]
        out = {'pred_logits': output_confid, 'pred_points': output_coord, 'offset': offset}
        if not self.aux_en or not self.training:
            return out
        else:
            raise NotImplemented  # still refinement, will be announced ASAP
            out['aux'] = None
            return out


class AnchorPoints(nn.Module):
    # Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
    # Single Layer Version
    def __init__(self, pyramid_levels=None, stride=None, row=3, line=3):
        super(AnchorPoints, self).__init__()
        self.pyramid_level = pyramid_levels  # default: 3
        if stride is None:
            self.stride = 2 ** self.pyramid_level  # default: 8
        else:
            self.stride = stride  # default 8
        self.row = row
        self.line = line

    def forward(self, image):
        image_shape = image.shape[2:]  # image shape: h,w
        image_shape = np.array(image_shape)
        image_shapes = (image_shape + self.stride - 1) // self.stride  # get downsample scale
        # get reference points for each level
        # each anchor block expand the number (row * line) of anchors
        anchor_points = self._generate_anchor_points(self.stride, row=self.row,
                                                     line=self.line)  # control the shift in the anchor block
        # anchor_map
        shifted_anchor_points = self._shift(image_shapes, self.stride, anchor_points)
        # get final anchor_map
        all_anchor_points = np.zeros((0, 2)).astype(np.float32)
        all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)
        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
        # send reference points to device
        if torch.cuda.is_available():
            return torch.from_numpy(all_anchor_points.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchor_points.astype(np.float32))

    def _generate_anchor_points(self, stride=16, row=3, line=3):
        # generate the reference points in grid layout
        row_step = stride / row
        line_step = stride / line

        shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
        shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        anchor_points = np.vstack((
            shift_x.ravel(), shift_y.ravel()
        )).transpose()
        return anchor_points

    def _shift(self, shape, stride, anchor_points):  # shape is feature map shape
        # shift the meta-anchor to get an acnhor points
        shift_x = (np.arange(0, shape[1]) + 0.5) * stride
        shift_y = (np.arange(0, shape[0]) + 0.5) * stride

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel()
        )).transpose()

        A = anchor_points.shape[0]  # num_of_points
        K = shifts.shape[0]  # num_of_pixel
        all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
        all_anchor_points = all_anchor_points.reshape((K * A, 2))
        return all_anchor_points


################### IFI part function ###################
class ifi_simfpn(nn.Module):
    def __init__(self, ultra_pe=False, pos_dim=40, sync_bn=False, num_anchor_points=4, num_classes=2, local=False,
                 unfold=False,
                 stride=1, learn_pe=False, require_grad=False, head_layers=[512, 256, 256], feat_num=4, feat_dim=256):
        # feat_num: # of encoder feats; num_anchor_points: line*row, num of queries each pixel;
        # num_classes: 2 {'regression':2, 'classifier':confidence},
        # ultra_pe/learn_pe: additional position encoding, pos_dim: additional position encoding dimension
        # local/unflod: impose the local feature information, head_layers: head layers setting, defaults=[512,256,256]
        # feat_dim: input_feats_dims
        super(ifi_simfpn, self).__init__()
        self.pos_dim = pos_dim
        self.ultra_pe = ultra_pe
        self.local = local
        self.unfold = unfold
        self.stride = stride
        self.learn_pe = learn_pe
        self.feat_num = feat_num
        self.feat_dim = feat_dim
        self.num_anchor_points = num_anchor_points
        self.regression_dims = 2  # fixed, coords
        self.num_classes = num_classes  # default: 2, confidence
        self.head_layers = head_layers
        norm_layer = nn.SyncBatchNorm if sync_bn else nn.BatchNorm1d  # nn.BatchNorm1d / nn.InstanceNorm2d

        # IFI.feat.encoder
        if learn_pe:
            print("learn_pe")
            for level in range(self.feat_num):
                self._update_property('pos' + str(level + 1), PositionEmbeddingLearned(self.pos_dim // 2))
        elif ultra_pe:
            print("ultra_pe")
            for level in range(self.feat_num):
                self._update_property('pos' + str(level + 1),
                                      SpatialEncoding(2, self.pos_dim, require_grad=require_grad))
            self.pos_dim += 2
        else:
            self.pos_dim = 2  # not use auxiliary position encoding. only (x, y)

        # Predict Heads
        in_dim = self.feat_num * (self.feat_dim + self.pos_dim)
        if unfold:
            in_dim = self.feat_num * (self.feat_dim * 9 + self.pos_dim)
        self.in_dim = in_dim

        confidence_head_list = []
        offset_head_list = []

        for ct, hidden_feature in enumerate(head_layers):
            if ct == 0:
                src_dim = in_dim
            else:
                src_dim = head_layers[ct - 1]
            confidence_head_list.append([nn.Conv1d(src_dim, hidden_feature, 1), norm_layer(hidden_feature), nn.ReLU()])
            offset_head_list.append([nn.Conv1d(src_dim, hidden_feature, 1), norm_layer(hidden_feature), nn.ReLU()])

        confidence_head_list.append(
            [nn.Conv1d(head_layers[-1], self.num_anchor_points * self.num_classes, 1), nn.ReLU()])
        offset_head_list.append([nn.Conv1d(head_layers[-1], self.num_anchor_points * 2, 1)])

        # build heads
        confidence_head_list = [item for sublist in confidence_head_list for item in sublist]
        offset_head_list = [item for sublist in offset_head_list for item in sublist]

        self.confidence_head = nn.Sequential(*confidence_head_list)
        self.offset_head = nn.Sequential(*offset_head_list)

    def forward(self, x, size, level=0, after_cat=False):
        h, w = size
        if not after_cat:
            if not self.local:
                if self.unfold:
                    x = F.unfold(x, 3, padding=1).view(x.shape[0], x.shape[1] * 9, x.shape[2], x.shape[3])
                rel_coord, q_feat = ifi_feat(x, [h, w])
                if self.ultra_pe:
                    rel_coord = eval('self.pos' + str(level))(rel_coord)
                elif self.learn_pe:
                    rel_coord = eval('self.pos' + str(level))(rel_coord, [1, 1, h, w])
                x = torch.cat([rel_coord, q_feat], dim=-1)
            else:
                if self.unfold:
                    x = F.unfold(x, 3, padding=1).view(x.shape[0], x.shape[1] * 9, x.shape[2], x.shape[3])
                rel_coord_list, q_feat_list, area_list = ifi_feat(x, [h, w], local=True, stride=self.stride)
                total_area = torch.stack(area_list).sum(dim=0)
                context_list = []
                for rel_coord, q_feat, area in zip(rel_coord_list, q_feat_list, area_list):
                    if self.ultra_pe:
                        rel_coord = eval('self.pos' + str(level))(rel_coord)
                    elif self.learn_pe:
                        rel_coord = eval('self.pos' + str(level))(rel_coord, [1, 1, h, w])
                    context_list.append(torch.cat([rel_coord, q_feat], dim=-1))
                ret = 0
                t = area_list[0]
                area_list[0] = area_list[3]
                area_list[3] = t
                t = area_list[1]
                area_list[1] = area_list[2]
                area_list[2] = t
                for conte, area in zip(context_list, area_list):
                    x = ret + conte * ((area / total_area).unsqueeze(-1))
            return x
        else:  # make output
            ############### for Conv1d ##############
            # offset regression
            offset = self.offset_head(x).view(x.shape[0], -1, h, w)
            offset = offset.permute(0, 2, 3, 1)  # # b, h, w, line*cow*2
            offset = offset.contiguous().view(x.shape[0], -1, 2)  # b, num_queries, 2

            # classifier
            confidence = self.confidence_head(x).view(x.shape[0], -1, h, w)
            confidence = confidence.permute(0, 2, 3, 1)  # # b, h, w, line*cow*num_classes
            confidence = confidence.contiguous().view(x.shape[0], -1,
                                                      self.num_classes)  # b, num_queries, self.num_classes
            return offset, confidence

    def _update_property(self, property, value):
        setattr(self, property, value)


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=128):
        super().__init__()
        self.row_embed = nn.Embedding(200, num_pos_feats)
        self.col_embed = nn.Embedding(200, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x, shape):
        # input: x, [b, N, 2]
        # output: [b, N, C]

        # h = w = int(np.sqrt(x.shape[1]))
        h, w = shape[2], shape[3]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1).view(x.shape[0], h * w, -1)
        return pos


class SpatialEncoding(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 sigma=6,
                 cat_input=True,
                 require_grad=False, ):

        super().__init__()
        assert out_dim % (2 * in_dim) == 0, "dimension must be dividable"

        n = out_dim // 2 // in_dim
        m = 2 ** np.linspace(0, sigma, n)
        m = np.stack([m] + [np.zeros_like(m)] * (in_dim - 1), axis=-1)
        m = np.concatenate([np.roll(m, i, axis=-1) for i in range(in_dim)], axis=0)
        self.emb = torch.FloatTensor(m)
        if require_grad:
            self.emb = nn.Parameter(self.emb, requires_grad=True)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma
        self.cat_input = cat_input
        self.require_grad = require_grad

    def forward(self, x):
        if not self.require_grad:
            self.emb = self.emb.to(x.device)
        y = torch.matmul(x, self.emb.T)
        if self.cat_input:
            return torch.cat([x, torch.sin(y), torch.cos(y)], dim=-1)
        else:
            return torch.cat([torch.sin(y), torch.cos(y)], dim=-1)


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):  # shape: h,w
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    # print("Coord:", ret.size())
    return ret


def ifi_feat(res, size, stride=1, local=False):
    '''
    res is input feature map, size is target scale (h, w)
    rel_coord is target mapping with feature coords.
    ex. target size 64*64 will find the nearest feature coords.
    '''
    # local is define local patch: 3*3 mapping near by center point.
    bs, hh, ww = res.shape[0], res.shape[-2], res.shape[-1]
    h, w = size
    coords = (make_coord((h, w)).cuda().flip(-1) + 1) / 2
    # coords = (make_coord((h,w)).flip(-1) + 1) / 2
    coords = coords.unsqueeze(0).expand(bs, *coords.shape)
    coords = (coords * 2 - 1).flip(-1)

    feat_coords = make_coord((hh, ww), flatten=False).cuda().permute(2, 0, 1).unsqueeze(0).expand(res.shape[0], 2,
                                                                                                  *(hh, ww))

    if local:
        vx_list = [-1, 1]
        vy_list = [-1, 1]
        eps_shift = 1e-6
        rel_coord_list = []
        q_feat_list = []
        area_list = []
    else:
        vx_list, vy_list, eps_shift = [0], [0], 0
    rx = stride / h
    ry = stride / w

    for vx in vx_list:
        for vy in vy_list:
            coords_ = coords.clone()
            coords_[:, :, 0] += vx * rx + eps_shift
            coords_[:, :, 1] += vy * ry + eps_shift
            coords_.clamp_(-1 + 1e-6, 1 - 1e-6)
            q_feat = F.grid_sample(res, coords_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)
            q_coord = F.grid_sample(feat_coords, coords_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:,
                      :, 0, :].permute(0, 2, 1)
            rel_coord = coords - q_coord
            rel_coord[:, :, 0] *= hh  # res.shape[-2]
            rel_coord[:, :, 1] *= ww  # res.shape[-1]
            if local:
                rel_coord_list.append(rel_coord)
                q_feat_list.append(q_feat)
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                area_list.append(area + 1e-9)
    if not local:
        return rel_coord, q_feat
    else:
        return rel_coord_list, q_feat_list, area_list


def build_ifi_head(cfg):
    return IFI_Decoder_Model(feat_layers=cfg.ifi_dict['feat_layers'],
                             num_classes=cfg.ifi_dict['num_classes'],
                             num_anchor_points=cfg.ifi_dict['num_anchor_points'],
                             line=cfg.ifi_dict['line'], row=cfg.ifi_dict['row'],
                             inner_planes=cfg.num_channels,
                             sync_bn=cfg.ifi_dict['sync_bn'],
                             require_grad=cfg.ifi_dict['require_grad'],
                             head_layers=cfg.ifi_dict['head_layers'],
                             pos_dim=cfg.ifi_dict['pos_dim'],
                             ultra_pe=cfg.ifi_dict['ultra_pe'],
                             learn_pe=cfg.ifi_dict['learn_pe'],
                             unfold=cfg.ifi_dict['unfold'],
                             local=cfg.ifi_dict['local'],
                             stride=cfg.ifi_dict['stride'],
                             AUX_EN=cfg.aux_en,
                             AUX_NUMBER=cfg.aux_number,
                             AUX_RANGE=cfg.aux_range,
                             AUX_kwargs=cfg.aux_kwargs,
                             )
