from typing import Tuple

import torch
from torch import nn
from mmcv.runner import force_fp32
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule
from ..builder import VTRANSFORMS
from method.ops import bev_pool

def gen_dx_bx(xbound, ybound, zbound):
    # dx:Resolution, bx:Bound, nx:Num_Grids
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx

class BaseTransform(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        # resolution, boundary, num of grids
        self.dx, self.bx, self.nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)

        self.C = out_channels
        self.frustum = self.create_frustum()
        # self.D = D moved into creat_frustum for convenience
        self.fp16_enabled = False

    @force_fp32()
    def create_frustum(self):
        """
        determine xyz locations in camera frame (Df, Hf, Wf, 3),
        each pseudo point will be assigned to the nearest pillar
        and generate the 'splatted' bev feature.
        Df * Hf * Wf = Di * Hi * Wi is the only requirement in sampling
        ### Experimental
        modified to generate slightly-more-uniform points.
        that is, directly sample every image feature pixel
        and adjust Df accordingly
        """
        Hi, Wi = self.image_size
        Hf, Wf = self.feature_size
        dmin, dmax, dx = self.dbound
        D = int((dmax - dmin) / dx)
        Df = (D*Hi*Wi)//(256*Hf*Wf)
        ds = (
            torch.linspace(dmin, dmax-dx, Df, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, Hf, Wf)
        )
        # uniformly sample (Hf,Wf) locations on (Hi,Wi)
        xs = (
            torch.linspace(0, Wi - 1, Wf, dtype=torch.float)
            .view(1, 1, Wf)
            .expand(Df, Hf, Wf)
        )
        ys = (
            torch.linspace(0, Hi - 1, Hf, dtype=torch.float)
            .view(1, Hf, 1)
            .expand(Df, Hf, Wf)
        )

        self.D = D
        # Df, Hf, Wf, 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    @force_fp32()
    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        post_rots,
        post_trans,
        **kwargs,
    ):
        B, N, _ = camera2lidar_trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )
        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x):
        raise NotImplementedError

    @force_fp32()
    def bev_pool(self, geom_feats, x):
        self.bx = self.bx.to(geom_feats.device)
        self.dx = self.dx.to(geom_feats.device)
        self.nx = self.nx.to(geom_feats.device)
        
        # pseudo point cloud (B*N*D*Hi*Wi, C)
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long() # voxel 坐标
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1) # (Nprime, 4), [x y z batch_ix]
        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        # input: feats, coords, B, D, H, W
        # output: (B, C, Z, X, Y)
        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])
        x = x.transpose(dim0=-2, dim1=-1) # performs better

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1) # (B, C*Z, X, Y), Z=1
        return final

    @force_fp32()
    def forward(
        self,
        imgs,
        geometries=None,
        **kwargs
        # added kwargs for consistency with other methods
    ):
        """
        Args:
            imgs: list(Torch.Tensor) [[B, N, C, H, W]]
            geometries (dict):
                "camera2lidar": 4x4 rigid transform matric, [B, N, 4, 4]
                "camera_intrinsics": 4x4 rigid transform matric, [B, N, 4, 4]
                "img_aug_matrix": 4x4 rigid transform matric, [B, N, 4, 4]
        """
        intrins = geometries["camera_intrinsics"][..., :3, :3]
        post_rots = geometries["img_aug_matrix"][..., :3, :3]
        post_trans = geometries["img_aug_matrix"][..., :3, 3]
        camera2lidar_rots = geometries["camera2lidar"][..., :3, :3]
        camera2lidar_trans = geometries["camera2lidar"][..., :3, 3]

        # print('---------------------LSSVT--------------------')
        # create feature frustum (Df, Hf, Wf, 3)
        # transform sample locations (B, N, Df, Hf, Wf, 3)
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
        )
        # generate pseudo point cloud (B, N, Di, Hi, Wi, C)
        x = self.get_cam_feats(imgs)
        # print('get_feats: ', x.size())
        # generate bev feature (B, C, h, w)
        # TODO: h,w = detect_range_(xy) / vt_grid_res[:2]
        x = self.bev_pool(geom, x)
        # print('bev_shape: ', x.shape)
        return x 

@VTRANSFORMS.register_module()
class LSSTransform(BaseTransform):
    def __init__(
        self,
        in_channels,
        out_channels,
        image_size,
        feature_size,
        # for arg define convenience
        pc_range=[-51.2, -51.2, -3, 51.2, 51.2, 5],
        pc_grid=[0.8, 0.8, 8],
        depth_step=[1.0, 60.0, 1.0],
        downsample=1,
        input_index=0,
    ) -> None:
        '''
        input = (B, N, C, Hi, Wi) where Hi,Wi = image_size/16;
        geometries = (B, N, D, Hf, Wf, 3) where Hf,Wf = feature_size;
        generated pseudo point cloud = (B, N, D, Hi, Wi, C);
        project (D, Hi, Wi) points on (Df, Hf, Wf) geom coordinates;
        then bev pool (B, C, Z, X, Y) and collapse Z to (B, C, X, Y),
        where (X,Y,Z) = (pc_range[3:] - pc_range[:3]) / pc_grid
        '''
        # Df = (D * Hi * Wi) / (Hf * Wf) must be integer
        assert (int((depth_step[1]-depth_step[0])/depth_step[2]) * \
                image_size[0]*image_size[1]//256) \
                % feature_size[0]*feature_size[1] == 0
        # bound = (min, max, step) on a certain axis
        xyzbounds = [(pc_range[axis], pc_range[axis+3], pc_grid[axis]) for axis in range(3)]
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            # NOTE: x,y <--> 0,1 is unconfirmed
            xbound=xyzbounds[0],
            ybound=xyzbounds[1],
            zbound=xyzbounds[2],
            dbound=depth_step,
        )
        self.input_index=input_index
        self.depthnet = nn.Conv2d(in_channels, self.D + self.C, 1) # 只有一层卷积？
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    @force_fp32()
    def get_cam_feats(self, x):
        # get top feature only
        if isinstance(x, list):
            x = x[self.input_index]
        B, N, C, Hi, Wi = x.shape

        x = x.view(B * N, C, Hi, Wi)

        x = self.depthnet(x) # (B*N, D+C, H, W)
        depth = x[:, : self.D].softmax(dim=1) # (B*N, D, H, W), softmax 将取值 norm 到 0~1 之间, 预测深度的置信度
        
        # (B*N, 1, D, H, W) * (B*N, C, 1, H, W) -> (B*N, C, D, H, W)
        # weighted features with confidence of predicted depth
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, Hi, Wi)
        x = x.permute(0, 1, 3, 4, 5, 2) # (B, N, D, H, W, C)
        return x

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        x = self.downsample(x)
        # print('out: ', x.shape)
        return x
