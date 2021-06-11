from random_eraser import get_random_eraser
import numpy as np
from natsort import natsorted
from os import listdir,mkdir,makedirs
from os.path import join,isdir 
from PIL import Image
import time
import shutil 
import fnmatch
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img


width_shift=0.05
channel_shift=30
rotate=0
zoom_range=0.05
batch_size = 1
augment_size=20
img_format='png'
aug_precent=0.5 # the potion of origianl images will be augmented


root_path=r"C:\Users\Kinect\Desktop\patches\patches-random-mixed-close"
output_path=root_path+'-aug-'+str(aug_precent)



data_gen_args =  dict(width_shift_range=width_shift, 
                                    height_shift_range=0,
                                    rotation_range=rotate,
                                    fill_mode = "nearest",
                                    zoom_range=zoom_range,
                                    horizontal_flip=True,
                                    featurewise_center=False, 
                                    featurewise_std_normalization=False,
                                    zca_whitening=False,
                                    channel_shift_range=channel_shift,
                                    preprocessing_function=get_random_eraser(s_h=0.2, pixel_level=True))
rgb_datagen =ImageDataGenerator(**data_gen_args)


t1 = time.time()


if isdir(output_path):
    print('rm folder: {}'.format(output_path))
    shutil.rmtree(output_path,ignore_errors=True)

path_lists=listdir(root_path)
path_lists=natsorted(path_lists)

seeds=[]
for p in path_lists:

    if p.find('result')!=-1 or p.find('.ini')!=-1:
        continue

    print('working on this: {0}'.format(p))

    seed = np.random.randint(100)
    while seed in seeds:
        seed = np.random.randint(100)
    seeds.append(seed)
    print('random seed: {0}'.format(seed))

    path = join(root_path,p)
    print(p)
    

    patch_list = listdir(path)
    patch_list = natsorted(patch_list)

    


    for s in patch_list:
        window_path=join(path,s,'window')
        bottles_list=listdir(window_path)

        for b in bottles_list:
            bottle_path=join(window_path,b)
            files=natsorted(listdir(bottle_path))

            dest=join(output_path,p,s,'window',b)
            makedirs(dest)


            #generate random int
            tmp= np.arange( len(files) )
            aug_size=int(aug_precent*tmp.shape[0])
            np.random.shuffle(tmp)
            rints=np.sort(tmp[:aug_size])

            for ind,f in enumerate(files):
                if f.find('.txt')!=-1:
                    shutil.copyfile(join(bottle_path,f),join(dest,f))
                elif f.find('.'+img_format)!=-1 and ind in rints:
                    rgb_img=Image.open(join(bottle_path,f))
                    rgb_img = img_to_array(rgb_img)
                    rgb_img = rgb_img.reshape((1,) + rgb_img.shape)
                    rgb_generator = rgb_datagen.flow(rgb_img ,batch_size=1,
                              save_to_dir=dest, save_prefix='rgb-{}'.format(ind),
                              save_format=img_format)
                    j = 0
                    for t in rgb_generator:
                        j += 1

                        if j > augment_size-1:
                            break



t2=time.time()

print('total time: {0}'.format(t2-t1))
