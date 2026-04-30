import torch
import torch.nn as nn
import torch.nn.functional as F
from .heads import build_task_head


# Scales aligned with official YOLOv12 configs (depth, width, max_channels).
YOLOV12_SCALES = {
    "n": dict(depth=0.50, width=0.25, max_channels=1024),
    "s": dict(depth=0.50, width=0.50, max_channels=1024),
    "m": dict(depth=0.50, width=1.00, max_channels=512),
    "l": dict(depth=1.00, width=1.00, max_channels=512),
    "x": dict(depth=1.00, width=1.50, max_channels=512),
}

def make_divisible(x, divisor=8):
    return int((x + divisor / 2) // divisor * divisor)

# ---------------------------
# 1. Area Attention (A)
# ---------------------------
class AreaAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"AreaAttention dim {dim} must be divisible by num_heads {num_heads}.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # Flatten to [B, Tokens, C]
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        qkv = self.qkv(x_flat).reshape(B, H * W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Use PyTorch's native FlashAttention-ready function
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.proj(out)
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

# ---------------------------
# 2. A2C2f (Attention-Augmented C2f)
# ---------------------------
class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p or k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, k=(3, 3)):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1, g=g)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class AAttn(nn.Module):
    def __init__(self, dim, num_heads, area=1):
        super().__init__()
        self.area = area
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        all_head_dim = self.head_dim * self.num_heads
        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape

        q, k = qk.split([C, C], dim=2)
        q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
        k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
        v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1))
        x = x.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return self.proj(x + pp)


class ABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__()
        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class A2C2f(nn.Module):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)
        if c_ % 32 != 0:
            raise ValueError("Dimension of ABlock must be a multiple of 32.")
        num_heads = c_ // 32
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)
        init_values = 0.01
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) if a2 else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * out
        return out

# ---------------------------
# 3. Backbone (Updated for v12)
# ---------------------------
class Backbone(nn.Module):
    def __init__(self, model_size="l"):
        super().__init__()
        s = YOLOV12_SCALES[model_size]
        w, d, mc = s["width"], s["depth"], s["max_channels"]

        def c(ch):
            return make_divisible(min(ch, mc) * w)

        def n(rep):
            return max(1, int(round(rep * d)))

        self.stem = Conv(3, c(64), 3, 2)
        self.p2 = Conv(c(64), c(128), 3, 2, 1, g=2)
        self.c3k2_1 = C3k2(c(128), c(256), n(2), c3k=False, e=0.25)
        self.p3 = Conv(c(256), c(256), 3, 2, 1, g=4)
        self.c3k2_2 = C3k2(c(256), c(512), n(2), c3k=False, e=0.25)
        self.p4 = Conv(c(512), c(512), 3, 2)
        self.a2c2f_1 = A2C2f(c(512), c(512), n(4), a2=True, area=4)
        self.p5 = Conv(c(512), c(1024), 3, 2)
        self.a2c2f_2 = A2C2f(c(1024), c(1024), n(4), a2=True, area=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.p2(x)
        x = self.c3k2_1(x)
        p3 = self.p3(x)
        p3 = self.c3k2_2(p3)
        p4 = self.p4(p3)
        p4 = self.a2c2f_1(p4)
        p5 = self.p5(p4)
        p5 = self.a2c2f_2(p5)
        return p3, p4, p5

# ---------------------------
# 5. YOLOv12 Wrapper (Matches YOLOIPC8 Behavior)
# ---------------------------
class YOLOv12(nn.Module):
    def __init__(
        self,
        model_size="l",
        num_classes=80,
        reg_max=16,
        task: str = "detection",
        num_masks: int = 32,
        num_protos: int = 32,
    ):
        super().__init__()
        self.task = str(task).lower()
        self.backbone = Backbone(model_size)

        s = YOLOV12_SCALES[model_size]
        w, mc = s["width"], s["max_channels"]

        def c(ch):
            return make_divisible(min(ch, mc) * w)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.c3 = c(256)
        self.c4 = c(512)
        self.c5 = c(1024)

        # Neck + head as per official config
        self.head1 = A2C2f(self.c5 + self.c4, self.c4, n=2, a2=False, area=-1)
        self.head2 = A2C2f(self.c4 + self.c3, self.c3, n=2, a2=False, area=-1)
        self.down1 = Conv(self.c3, self.c3, 3, 2)
        self.head3 = A2C2f(self.c3 + self.c4, self.c4, n=2, a2=False, area=-1)
        self.down2 = Conv(self.c4, self.c4, 3, 2)
        self.head4 = C3k2(self.c4 + self.c5, self.c5, n=2, c3k=True)

        self.task_head = build_task_head(
            task=self.task,
            ch=[self.c3, self.c4, self.c5],
            nc=num_classes,
            reg_max=reg_max,
            decoupled=True,
            num_masks=num_masks,
            num_protos=num_protos,
        )

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        x = self.up(p5)
        x = torch.cat([x, p4], dim=1)
        p4h = self.head1(x)
        x = self.up(p4h)
        x = torch.cat([x, p3], dim=1)
        p3h = self.head2(x)
        x = self.down1(p3h)
        x = torch.cat([x, p4h], dim=1)
        p4o = self.head3(x)
        x = self.down2(p4o)
        x = torch.cat([x, p5], dim=1)
        p5o = self.head4(x)
        return self.task_head([p3h, p4o, p5o])

    @property
    def detect(self):
        return self.task_head
