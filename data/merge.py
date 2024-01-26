import os
import shutil
from tqdm import tqdm

DATA_DIR = "/dcs/large/u2009169/"
CIDIAN_DIR = DATA_DIR + "seal-script-images/"
SHUOWEN_DIR = DATA_DIR + "seal-script-shuowen-max/"


# rename every file in shuowen
dirs = [d.name for d in os.scandir(SHUOWEN_DIR) if d.is_dir()]

def move(dirno):
    fullPath = SHUOWEN_DIR + dirno + "/"
    destPath = CIDIAN_DIR + dirno + "/"
    for image in os.listdir(fullPath):
        parts = image.split(".")
        newPath = parts[0] + "_sw." + parts[1]  # turns 100_1.png to 100_1_sw.png so no clashes
        os.rename(fullPath + image, fullPath + newPath)
        shutil.copy(fullPath + newPath, destPath + newPath)
        
for d in tqdm(dirs):
    move(d)