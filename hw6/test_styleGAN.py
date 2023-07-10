import subprocess
import os

cmd_str = "stylegan2_pytorch --generate --load-from 150 --image-size 64 --attn-layers [1,2] --num_image_tiles 1 --num_generate 1000 --trunc-psi 0.5"
subprocess.run(cmd_str, shell=True)

folder_path = './results/default/'

files = os.listdir(folder_path)

renamed_files = 0

for file_name in files:
    old_file_path = os.path.join(folder_path, file_name)

    if renamed_files >= 1000:
        os.remove(old_file_path)
        continue

    if "ema" in file_name:
        new_file_name = str(renamed_files + 1) + '.jpg'

        new_file_path = os.path.join(folder_path, new_file_name)

        os.rename(old_file_path, new_file_path)

        renamed_files += 1
    else:
        os.remove(old_file_path)

