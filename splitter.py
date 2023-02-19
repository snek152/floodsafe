import os
import shutil

import os
import shutil

#splits dataset 80 - 10 - 10 --> training testing validation

source_dir = 'no_ar_images/'
dest_dir = 'train_images/'
dest1_dir = 'test_images/'
dest2_dir = 'val_images/'

count = 0
index = 0
index1 = 0

for filename in os.listdir(source_dir):
    if count < 924:
        if filename.endswith('.png'):  
            src_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest_dir, f'no_vapor_{count}.png')
            shutil.copy(src_path, dest_path)
            count += 1
    elif count < 1040:
        if filename.endswith('.png'):  
            src_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest1_dir, f'no_vapor_{index}.png')
            shutil.copy(src_path, dest_path)
            count += 1
            index += 1
    else:
        if filename.endswith('.png'):  
            src_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest2_dir, f'no_vapor_{index1}.png')
            shutil.copy(src_path, dest_path)
            count += 1
            index1 += 1


# source_dir = 'images/'
# dest_dir = 'train_images/'
# dest1_dir = 'test_images/'
# dest2_dir = 'val_images/'

# count = 0
# index = 0
# index1 = 0

# for filename in os.listdir(source_dir):
#     if count < 924:
#         if filename.endswith('.png'):  
#             src_path = os.path.join(source_dir, filename)
#             dest_path = os.path.join(dest_dir, f'vapor_{count}.png')
#             shutil.copy(src_path, dest_path)
#             count += 1
#     elif count < 1040:
#         if filename.endswith('.png'):  
#             src_path = os.path.join(source_dir, filename)
#             dest_path = os.path.join(dest1_dir, f'vapor_{index}.png')
#             shutil.copy(src_path, dest_path)
#             count += 1
#             index += 1
#     else:
#         if filename.endswith('.png'):  
#             src_path = os.path.join(source_dir, filename)
#             dest_path = os.path.join(dest2_dir, f'vapor_{index1}.png')
#             shutil.copy(src_path, dest_path)
#             count += 1
#             index1 += 1



    