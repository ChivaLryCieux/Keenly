from PIL import Image
import numpy as np
import sys

path = sys.argv[1] if len(sys.argv) > 1 else r'out\showcase_grid.png'
a = np.asarray(Image.open(path).convert('RGB'), dtype=np.float64) / 255.0
tile = int(sys.argv[2]) if len(sys.argv) > 2 else 800
gap = int(sys.argv[3]) if len(sys.argv) > 3 else 16
names = ['ice/facet', 'mono/prism', 'ember/prism', 'ice/frost', 'mono/frost',
         'dusk/liquid', 'cinema/liquid', 'ember/facet', 'cinema/warp']
print('cell            lum   std   edge   sat   histogram (8 bins, %)')
for i in range(9):
    cx = gap + (i % 3) * (tile + gap)
    cy = gap + (i // 3) * (tile + gap)
    c = a[cy:cy + tile, cx:cx + tile]
    lum = c @ [0.2126, 0.7152, 0.0722]
    gx = np.abs(np.diff(lum, axis=1)).mean()
    gy = np.abs(np.diff(lum, axis=0)).mean()
    sat = (c.max(2) - c.min(2)).mean()
    h = (np.histogram(lum, bins=8, range=(0, 1))[0] / lum.size * 100).round(0).astype(int).tolist()
    print('%-13s %5.2f %5.2f %6.4f %5.2f   %s' % (names[i], lum.mean(), lum.std(), (gx + gy) / 2, sat, h))