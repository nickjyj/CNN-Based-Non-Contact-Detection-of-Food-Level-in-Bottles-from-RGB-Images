import numpy as np
from natsort import natsorted
from sklearn.model_selection import train_test_split
from collections import defaultdict
from os import listdir,makedirs
from os.path import join,isdir,isfile,expanduser
import utils,nn_models
import time
from keras.utils import to_categorical # utilities for one-hot encoding of ground truth values
from keras.callbacks import EarlyStopping,ModelCheckpoint
import re
import random_eraser
from keras.preprocessing.image import ImageDataGenerator
import csv


batch_size = 32
num_epochs = 10
num_outputs=0
num_classes=5
num_color_channels=3
load_liquid_ids=False
is_binary_class=False
is_save_imgs=True
data_augmentation = False


fieldnames = ['leave_one_bottle', 'last_epoch', 'best_val']
input_size=[120,60]
desktop_path=expanduser("~/Desktop/patches")
input_paths=['patches-kens-bare-close',
           'patches-kraft-bare-close',
           'patches-random-bare-close',]

root_dirs=[join(desktop_path,i) for i in input_paths]

aug_paths=[i + '-aug-0.5' for i in input_paths]
aug_paths=[join(desktop_path,i) for i in aug_paths]

output_prefix=join(desktop_path, 'results_bare_10_bottles_10th',
                'results_ch'+str(num_color_channels)+
                '_e'+str(num_epochs)+'_k'+str(nn_models.kernel_size)+'_')

output_dir=output_prefix + 'aug' if data_augmentation else output_prefix + 'no-aug'
output_dir+='-b' if is_binary_class else ''



t1 = time.time()

load_gray=False
if num_color_channels==1:
    load_gray=True


if data_augmentation:
    print('loading augmentation data')
    dict_aug_win={}
    for p in aug_paths:
        tmp_win,_=utils.load_all_patches(p,input_size,load_background=False, is_gray=load_gray,binary_class=is_binary_class,load_liquid_ids=load_liquid_ids)
        dict_aug_win={**dict_aug_win,**tmp_win}

dict_win={}
dict_b={}

for p in root_dirs:
    tmp_win,tmp_b=utils.load_all_patches(p,input_size,binary_class=is_binary_class,is_gray=load_gray,load_liquid_ids=load_liquid_ids)
    dict_win={**dict_win,**tmp_win}
    dict_b={**dict_b,**tmp_b}


if not isdir(output_dir):
    makedirs(output_dir)

if num_classes!=0:
    num_outputs=1


default_val=np.empty((0,input_size[0],input_size[1],num_color_channels),dtype=np.uint8)
default_labels=np.empty((0,num_outputs))

patches_b_all=default_val.copy()
labels_b_all=default_labels.copy()

'''
# cat the backgrounds
for k in dict_b:
    patches_b=dict_b[k][0]

    labels_b=dict_b[k][1][:,:num_outputs]


    patches_b_all=np.append(patches_b_all,patches_b,axis=0)
    labels_b_all=np.append(labels_b_all,labels_b,axis=0)
'''


# group by id of bottle
if not load_liquid_ids:
    dict_win_type=defaultdict(lambda: [default_val.copy(),default_labels.copy()])
else:
    dict_win_type=defaultdict(lambda: [default_val.copy(),default_labels.copy(),default_labels.copy()])

for k in dict_win:
    patches_win=dict_win[k][0]
    labels_win=dict_win[k][1]       # [levels, ids, liquid_ids]
    ids=labels_win[:,1]

    uniq=np.unique(ids)
    for n in uniq:
        mask=ids==n
        labels=labels_win[mask][:,:num_outputs]
        patches=patches_win[mask]

        if not isinstance(n,str):
            n=str(n)
        patches_tmp=np.append(dict_win_type[n][0],patches,axis=0)
        labels_tmp=np.append(dict_win_type[n][1],labels,axis=0)

        if load_liquid_ids:
            liquid_ids=labels_win[mask][:,2]
            liquid_ids_tmp=np.append(dict_win_type[n][2],liquid_ids[...,np.newaxis],axis=0)
            dict_win_type[n]=[patches_tmp,labels_tmp,liquid_ids_tmp]
        else:
            dict_win_type[n]=[patches_tmp,labels_tmp]


#group by id
if data_augmentation:
    if not load_liquid_ids:
        dict_win_aug_type=defaultdict(lambda: [default_val.copy(), default_labels.copy()])
    else:
        dict_win_aug_type=defaultdict(lambda: [default_val.copy(), default_labels.copy(), default_labels.copy()])
    for k in dict_aug_win:
        patches_win=dict_aug_win[k][0]
        labels_win=dict_aug_win[k][1]
        ids=labels_win[:,1]

        uniq=np.unique(ids)
        for n in uniq:
            mask=ids==n
            labels=labels_win[mask][:,:num_outputs]
            patches=patches_win[mask]
            
            if not isinstance(n,str):
                n=str(n)
            patches_tmp=np.append(dict_win_aug_type[n][0],patches,axis=0)
            labels_tmp=np.append(dict_win_aug_type[n][1],labels,axis=0)

            if load_liquid_ids:
                liquid_ids=labels_win[mask][:,2]
                liquid_ids_tmp=np.append(dict_win_aug_type[n][2],liquid_ids[...,np.newaxis],axis=0)
                dict_win_aug_type[n]=[patches_tmp,labels_tmp,liquid_ids_tmp]
            else:
                dict_win_aug_type[n]=[patches_tmp,labels_tmp]


# test one view point
for k in dict_win_type:
    if float(k)!=10:
        continue
    print('\ntest {}'.format(k))

    X_test=dict_win_type[k][0]
    Y_test=dict_win_type[k][1]

    if load_liquid_ids:
        liquid_ids_test=np.unique(dict_win_type[k][2])

    X_train=default_val.copy()
    Y_train=default_labels.copy()
    for k2 in dict_win_type:
        if k2 == k:
            continue
        print('train {}'.format(k2))

        
        if load_liquid_ids:
            liquid_ids_tmp=dict_win_type[k2][2]

            if len(liquid_ids_tmp.shape)>1:
                liquid_ids_tmp=np.squeeze(liquid_ids_tmp,axis=-1)

            mask=np.ones(liquid_ids_tmp.shape,dtype=bool)
            for i in liquid_ids_test:
                mask = np.logical_and(mask,liquid_ids_tmp!=i)

            left_liquid_ids=np.unique(liquid_ids_tmp[mask])
            print('train liquid: {}'.format(left_liquid_ids))

            X_train=np.append(X_train,dict_win_type[k2][0][mask],axis=0)
            Y_train=np.append(Y_train,dict_win_type[k2][1][mask],axis=0)
        else:
            X_train=np.append(X_train,dict_win_type[k2][0],axis=0)
            Y_train=np.append(Y_train,dict_win_type[k2][1],axis=0)

        if data_augmentation:

            if load_liquid_ids:
                liquid_ids_tmp=dict_win_aug_type[k2][2]

                if len(liquid_ids_tmp.shape)>1:
                    liquid_ids_tmp=np.squeeze(liquid_ids_tmp,axis=-1)

                mask=np.ones(liquid_ids_tmp.shape,dtype=bool)
                for i in liquid_ids_test:
                    mask = np.logical_and(mask,liquid_ids_tmp!=i)

                X_train=np.append(X_train,dict_win_aug_type[k2][0][mask],axis=0)
                Y_train=np.append(Y_train,dict_win_aug_type[k2][1][mask],axis=0)
            else:
                X_train=np.append(X_train,dict_win_aug_type[k2][0],axis=0)
                Y_train=np.append(Y_train,dict_win_aug_type[k2][1],axis=0)
    
    '''
    # split 50-50 for backgrounds
    X_train_b, X_test_b, Y_train_b, Y_test_b = train_test_split(patches_b_all,labels_b_all,test_size=0.5)
    
    X_train=np.append(X_train,X_train_b,axis=0)
    Y_train=np.append(Y_train,Y_train_b,axis=0)

    X_test=np.append(X_test,X_test_b,axis=0)
    Y_test=np.append(Y_test,Y_test_b,axis=0)
    '''


    if is_binary_class:
        num_classes=2

    if num_classes!=0:
        Y_train=to_categorical(Y_train,num_classes)
        Y_test=to_categorical(Y_test,num_classes)


    X_train, X_cv, Y_train, Y_cv = train_test_split(X_train,Y_train,test_size=0.2)


    print('number of train samples: {0}'.format(X_train.shape[0]))
    print('number of cv samples: {0}'.format(X_cv.shape[0]))
    print('number of test samples: {0}'.format(X_test.shape[0]))

    if num_classes==0:
        model = nn_models.deep_cnn2_regressor(X_train.shape[1:],num_outputs)
    else:
        model = nn_models.deep_cnn2(X_train.shape[1:],num_classes)


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
    model.summary()


    #"weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    filepath = join(output_dir,k + '-weights.hdf5')
    checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True)  
    

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs,
                verbose=2, validation_data=(X_cv,Y_cv), callbacks=[checkpointer])


    score1 = model.evaluate(X_test, Y_test, verbose=2)
    print('test mse of final epoch: {0}'.format(score1[1]))

    model.load_weights(filepath)
    print("loaded weights from file")

    score2 = model.evaluate(X_test, Y_test, verbose=2)
    print('test mse of best val: {0}'.format(score2[1]))
  

    y_predict = model.predict(X_test)

    np.savetxt(join(output_dir,'{0}-outputs.txt'.format(k)),y_predict,fmt='%.4f',delimiter=',')
    np.savetxt(join(output_dir,'{0}-labels.txt'.format(k)),Y_test,fmt='%d',delimiter=',')


    # write out header if file not exists
    if not isfile(join(output_dir,'score.csv')):
        with open(join(output_dir,'score.csv'),'a',newline='') as f:
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    with open(join(output_dir,'score.csv'),'a',newline='') as f:
        fieldnames = ['leave_one_bottle', 'last_epoch', 'best_val']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        score_r1 = round(score1[1], 4)
        score_r2 = round(score2[1], 4)
    
        #s = "{}: {} {}\n".format(k,score_r1,score_r2)
        writer.writerow({fieldnames[0]: k, fieldnames[1]: score_r1, fieldnames[2]: score_r2})


    if is_save_imgs:
        # scale test data
        X_test = utils.scale_imarray(X_test)

        # save imgs
        print('saving test images...')
        output_imdir = join(output_dir,k + '-imgs')
        if not isdir(output_imdir):
            makedirs(output_imdir)
        utils.save_imarray(X_test,output_imdir,Y_test,y_predict,map='gray')

    del X_train, X_cv, X_test, Y_train, Y_cv, Y_test
    t2 = time.time()
    print('total time: {0}'.format(t2 - t1))

# write out new line
with open(join(output_dir,'score.csv'),'a',newline='') as f:    
    writer = csv.DictWriter(f,fieldnames=fieldnames)
    writer.writerow({})