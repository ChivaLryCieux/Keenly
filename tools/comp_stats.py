from PIL import Image
import numpy as np
import sys

def metrics(path, name):
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.float64) / 255.0
    lum = a @ [0.2126, 0.7152, 0.0722]
    h, w = lum.shape
    by, bx = h // 8, w // 8
    blocks = lum[:by * 8, :bx * 8].reshape(8, by, 8, bx).mean(axis=(1, 3))
    comp = blocks.std()          # composition: light/dark layout variance
    fine = lum.std() - comp      # rough texture-vs-composition split
    print('%-22s lum %.3f  std %.3f  composition %.4f  (block means std)' %
          (name, lum.mean(), lum.std(), comp))

for p, n in zip(sys.argv[1::2], sys.argv[2::2]):
    metrics(p, n)