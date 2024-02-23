from glob import glob
import os
import shutil

FROM_DIR = '/dcs/20/u2002183/cs409-CCP-1/source/'
TO_DIR = '/dcs/20/u2002183/cs409-CCP-1/seal-script-images/'

for root, files, name in os.walk(FROM_DIR):
    if root != FROM_DIR:
        print(root)

        first = [x for x in name if '_1.png' in x][0]
        folder = first.split('_')[0]
        dest = TO_DIR + folder + '/'
        num_dest_files = len([name for name in os.listdir(dest) if (os.path.isfile(os.path.join(dest, name)) and '.png' in name)])
        dest_filepath = dest + folder + '_' + str(num_dest_files+1) + '.png'
        src_filepath = root + '/' + first
        shutil.copyfile(src_filepath, dest_filepath)

