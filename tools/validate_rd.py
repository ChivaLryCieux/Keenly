# -*- coding: utf-8 -*-
# Validate Gray-Scott parameter pairs survive with lattice-fixed seed init.
# Mirrors src/turing.cuh exactly (same hash, same update, same wrap).
import numpy as np

def hash01(x, y, seed):
    x = np.asarray(x, np.uint32); y = np.asarray(y, np.uint32)
    h = (x * np.uint32(0x9E3779B9) + y * np.uint32(0x85EBCA6B) + np.uint32(seed) * np.uint32(0xC2B2AE35)) & np.uint32(0xFFFFFFFF)
    h ^= h >> np.uint32(16); h = (h * np.uint32(0x7feb352d)) & np.uint32(0xFFFFFFFF)
    h ^= h >> np.uint32(15); h = (h * np.uint32(0x846ca68b)) & np.uint32(0xFFFFFFFF)
    h ^= h >> np.uint32(16)
    return (h >> np.uint32(8)).astype(np.float64) / 16777216.0

def lap(M):
    return (0.2*(np.roll(M,1,0)+np.roll(M,-1,0)+np.roll(M,1,1)+np.roll(M,-1,1))
          + 0.05*(np.roll(np.roll(M,1,0),1,1)+np.roll(np.roll(M,1,0),-1,1)+np.roll(np.roll(M,-1,0),1,1)+np.roll(np.roll(M,-1,0),-1,1)) - M)

def sim(f, k, res=512, iters=4000, seed=7):
    A = np.ones((res,res)); B = np.zeros((res,res))
    yy, xx = np.mgrid[0:res, 0:res]
    B[hash01(xx, yy, seed+991) > 0.9993] = 1.0
    for i in range(16):
        cx = hash01(i, 11, seed) * res; cy = hash01(i, 29, seed) * res
        r = 3.0 + 5.0*hash01(i, 47, seed)   # LATTICE units, resolution-independent
        dx = xx + 0.5 - cx; dy = yy + 0.5 - cy
        dx = np.where(dx > res/2, dx-res, dx); dx = np.where(dx < -res/2, dx+res, dx)
        dy = np.where(dy > res/2, dy-res, dy); dy = np.where(dy < -res/2, dy+res, dy)
        B[dx*dx+dy*dy < r*r] = 1.0
    for it in range(iters):
        la, lb = lap(A), lap(B)
        abb = A*B*B
        A = np.clip(A + (la - abb + f*(1-A)), 0, 1)
        B = np.clip(B + (0.5*lb + abb - (f+k)*B), 0, 1)
    return B

pairs = [("coral", 0.0545, 0.062), ("spots", 0.0367, 0.0649), ("worms", 0.022, 0.051),
         ("maze", 0.038, 0.061), ("stripes", 0.026, 0.051), ("chaos", 0.034, 0.056)]
for name, f, k in pairs:
    B = sim(f, k)
    raw = (B - 0.18) * 4.5
    raw = np.clip(raw, -1, 1)
    print('%-8s f %.4f k %.4f  B std %.4f max %.3f frac>0.1 %.3f | mapped std %.3f p05 %.2f p50 %.2f p95 %.2f' % (
        name, f, k, B.std(), B.max(), (B>0.1).mean(), raw.std(), np.percentile(raw,5), np.percentile(raw,50), np.percentile(raw,95)))