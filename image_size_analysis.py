import glob
import cv2
import numpy as np


sizes = []

for p in glob.glob(r"C:\Users\marcu\Documents\Year 3\Computer Vision\comp_vision\data\train\*\*"):
  img = cv2.imread(p)
  h, w, c = img.shape
  sizes.extend([h, w])


sizes_np = np.array(sizes)

print("Mean:", np.mean(sizes_np))
print("Max:", np.max(sizes_np))
print("Min:", np.min(sizes_np))
