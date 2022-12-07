data_root = 'diverse_occupation_test'

import os
image_paths = [os.path.join(data_root, file_path) for file_path in os.listdir(data_root)]

with open(data_root+'/base_words.txt', 'w') as fh:
    for image_path in image_paths:
        image_path_local = image_path.split('/')[-1]
        image_base_word = image_path_local.split('_')[1]
        if ' ' in image_base_word:
            print('warning:', image_base_word)
        print(image_path_local, image_base_word, file=fh)
