import glob
import os
import shutil
import random


TRAIN_VAL_DIR = r"C:\Users\marcu\Documents\Year 3\Computer Vision\Coursework 1\data\train_val"
TRAIN_OUTPUT_DIR = r"C:\Users\marcu\Documents\Year 3\Computer Vision\Coursework 1\data\train"
VAL_OUTPUT_DIR = r"C:\Users\marcu\Documents\Year 3\Computer Vision\Coursework 1\data\val"
VAL_RATIO = 0.2


def copyfile(file_path, src_folder):
  if not os.path.exists(src_folder):
    os.makedirs(src_folder)
    
  img_name = os.path.basename(file_path)
  out_path = os.path.join(src_folder, img_name)
  print(file_path + " -> " + out_path)
  shutil.copy(file_path, src_folder)


# Stratified random sampling
for class_dir in glob.glob(os.path.join(TRAIN_VAL_DIR, "*")):
  class_dir_name = os.path.basename(class_dir)

  class_images = glob.glob(os.path.join(class_dir, "*"))
  num_images = len(class_images)
  random.shuffle(class_images)

  val_images = class_images[:int(num_images*VAL_RATIO)]
  train_images = class_images[int(num_images*VAL_RATIO):]

  # Copy validation set files
  for class_img in val_images:
    copyfile(class_img, os.path.join(VAL_OUTPUT_DIR, class_dir_name))

  # Copy train set files
  for class_img in train_images:
    copyfile(class_img, os.path.join(TRAIN_OUTPUT_DIR, class_dir_name))
  

