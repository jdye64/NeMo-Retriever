# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA kernels for batched page rasterization. Compiled by CuPy at runtime."""

FILL_RECTS = r"""
extern "C" __global__
void fill_rects(
    unsigned char* fb, int N, int H, int W,
    const float* rects, int nrects)
{
    int rid = blockIdx.y;
    if (rid >= nrects) return;
    const float* r = rects + rid * 9;
    int page = (int)r[0];
    if (page < 0 || page >= N) return;
    int x0 = (int)floorf(r[1]);
    int y0 = (int)floorf(r[2]);
    int x1 = (int)ceilf(r[3]);
    int y1 = (int)ceilf(r[4]);
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > W) x1 = W;
    if (y1 > H) y1 = H;
    int bw = x1 - x0;
    int bh = y1 - y0;
    if (bw <= 0 || bh <= 0) return;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= bw * bh) return;
    int yy = y0 + t / bw;
    int xx = x0 + t % bw;
    float a = r[8];
    if (a <= 0.0f) return;
    size_t idx = (((size_t)page * H + yy) * W + xx) * 4;
    float ia = 1.0f - a;
    fb[idx + 0] = (unsigned char)(r[5] * 255.0f * a + fb[idx + 0] * ia);
    fb[idx + 1] = (unsigned char)(r[6] * 255.0f * a + fb[idx + 1] * ia);
    fb[idx + 2] = (unsigned char)(r[7] * 255.0f * a + fb[idx + 2] * ia);
    float oa = a + (fb[idx + 3] / 255.0f) * ia;
    fb[idx + 3] = (unsigned char)(oa * 255.0f);
}
"""

FILL_PATH = r"""
extern "C" __global__
void fill_path(
    unsigned char* fb, int N, int H, int W,
    int page, const float* edges, int nedges,
    float cr, float cg, float cb, float ca, int even_odd)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int npix = H * W;
    if (t >= npix || page < 0 || page >= N) return;
    int y = t / W;
    int x = t % W;
    float px = (float)x + 0.5f;
    float py = (float)y + 0.5f;
    int wind = 0;
    int cross = 0;
    for (int e = 0; e < nedges; ++e) {
        float x0 = edges[e * 4 + 0];
        float y0 = edges[e * 4 + 1];
        float x1 = edges[e * 4 + 2];
        float y1 = edges[e * 4 + 3];
        int above0 = y0 > py;
        int above1 = y1 > py;
        if (above0 == above1) continue;
        float tedge = (py - y0) / ((y1 - y0) + 1e-12f);
        float xs = x0 + tedge * (x1 - x0);
        if (xs >= px) {
            wind += (y1 > y0) ? 1 : -1;
            cross += 1;
        }
    }
    int inside = even_odd ? (cross & 1) : (wind != 0);
    if (!inside || ca <= 0.0f) return;
    size_t idx = (((size_t)page * H + y) * W + x) * 4;
    float ia = 1.0f - ca;
    fb[idx + 0] = (unsigned char)(cr * 255.0f * ca + fb[idx + 0] * ia);
    fb[idx + 1] = (unsigned char)(cg * 255.0f * ca + fb[idx + 1] * ia);
    fb[idx + 2] = (unsigned char)(cb * 255.0f * ca + fb[idx + 2] * ia);
    float oa = ca + (fb[idx + 3] / 255.0f) * ia;
    fb[idx + 3] = (unsigned char)(oa * 255.0f);
}
"""

BLIT = r"""
extern "C" __global__
void blit_image(
    unsigned char* fb, int N, int H, int W,
    int page, int x0, int y0, int x1, int y1,
    const unsigned char* src, int sH, int sW)
{
    int bw = x1 - x0;
    int bh = y1 - y0;
    if (bw <= 0 || bh <= 0) return;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= bw * bh) return;
    int yy = y0 + t / bw;
    int xx = x0 + t % bw;
    if (xx < 0 || yy < 0 || xx >= W || yy >= H || page < 0 || page >= N) return;
    float u = ((float)(xx - x0) + 0.5f) / (float)bw;
    float v = ((float)(yy - y0) + 0.5f) / (float)bh;
    int sx = (int)(u * sW);
    int sy = (int)(v * sH);
    if (sx < 0) sx = 0;
    if (sy < 0) sy = 0;
    if (sx >= sW) sx = sW - 1;
    if (sy >= sH) sy = sH - 1;
    size_t sidx = ((size_t)sy * sW + sx) * 4;
    float a = src[sidx + 3] / 255.0f;
    if (a <= 0.0f) return;
    size_t idx = (((size_t)page * H + yy) * W + xx) * 4;
    float ia = 1.0f - a;
    fb[idx + 0] = (unsigned char)(src[sidx + 0] * a + fb[idx + 0] * ia);
    fb[idx + 1] = (unsigned char)(src[sidx + 1] * a + fb[idx + 1] * ia);
    fb[idx + 2] = (unsigned char)(src[sidx + 2] * a + fb[idx + 2] * ia);
    float oa = a + (fb[idx + 3] / 255.0f) * ia;
    fb[idx + 3] = (unsigned char)(oa * 255.0f);
}
"""
