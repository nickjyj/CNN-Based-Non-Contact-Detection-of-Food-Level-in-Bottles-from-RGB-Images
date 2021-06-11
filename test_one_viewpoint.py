import numpy as np
from natsort import natsorted
from sklearn.model_selection import train_test_split

from os import listdir,mkdir
from os.path import join,isdir,isfile 
import utils,nn_models
import time
from keras.utils import np_utils,to_categorical # utilities for one-hot encoding of ground truth values
from keras.callbacks import EarlyStopping,ModelCheckpoint
import re


batch_size = 32
num_epochs = 1
num_outputs=2

input_size=[120,60]
root_dir=r'C:\Users\Kinect\Desktop\patches-kraft-close'
output_dir=join(root_dir,'results_e'+str(num_epochs)+'_k'+str(nn_models.kernel_size))
is_save_imgs=True



t1 = time.time()


dict_win,dict_b=utils.load_all_patches(root_dir,input_size)


if not isdir(output_dir):
    mkdir(output_dir)


dict_win_type={}
dict_b_type={}
# separate train and test datasets
for k in dict_win:

    patches_win=dict_win[k][0]
    patches_b=dict_b[k][0]

    labels_win=dict_win[k][1]
    labels_b=dict_b[k][1]

    m=re.split(r'-l[-0-9]+',k)
    p=m[0]
    print('viewPoint: {}, pattern: {}'.format(k,p))

    if p not in dict_win_type:
        dict_win_type[p]=[patches_win,labels_win]
        dict_b_type[p]=[patches_b,labels_b]
    else:
        patches_win_all=np.append(dict_win_type[p][0],patches_win,axis=0)
        patches_b_all=np.append(dict_b_type[p][0],patches_b,axis=0)

        labels_win_all=np.append(dict_win_type[p][1],labels_win,axis=0)
        labels_b_all=np.append(dict_b_type[p][1],labels_b,axis=0)

        dict_win_type[p]=[patches_win_all,labels_win_all]
        dict_b_type[p]=[patches_b_all,labels_b_all]


# test one view point
for k in dict_win_type:
    print('\ntest {}'.format(k))

    X_test=np.append(dict_win_type[k][0],dict_b_type[k][0],axis=0)
    Y_test=np.append(dict_win_type[k][1],dict_b_type[k][1],axis=0)

    X_train=None
    for k2 in dict_win_type:
        if k2 == k:
            continue
        print('train {}'.format(k2))
        if X_train is None:
            X_train=dict_win_type[k2][0]
            Y_train=dict_win_type[k2][1]

            X_train=np.append(X_train,dict_b_type[k2][0],axis=0)
            Y_train=np.append(Y_train,dict_b_type[k2][1],axis=0)
        else:
            X_train=np.append(X_train,dict_win_type[k2][0],axis=0)
            X_train=np.append(X_train,dict_b_type[k2][0],axis=0)

            Y_train=np.append(Y_train,dict_win_type[k2][1],axis=0)
            Y_train=np.append(Y_train,dict_b_type[k2][1],axis=0)


    X_train, X_cv, Y_train, Y_cv = train_test_split(X_train,Y_train,test_size=0.2)


    print('number of train samples: {0}'.format(X_train.shape[0]))
    print('number of cv samples: {0}'.format(X_cv.shape[0]))
    print('number of test samples: {0}'.format(X_test.shape[0]))

    model = nn_models.deep_cnn2_regressor(X_train.shape[1:],num_outputs)


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


    with open(join(output_dir,'score.txt'),'a') as f:
        score_r1 = round(score1[1], 4)
        score_r2 = round(score2[1], 4)
    
        s = "{}: {} {}\n".format(k,score_r1,score_r2)
        f.write(s)


    if is_save_imgs:
        # scale test data
        X_test = utils.scale_imarray(X_test)

        # save imgs
        print('saving test images...')
        output_imdir = join(output_dir,k + '-imgs')
        if not isdir(output_imdir):
            mkdir(output_imdir)
        utils.save_imarray(X_test,output_imdir,Y_test,y_predict,map='gray')

    t2 = time.time()
    print('total time: {0}'.format(t2 - t1))
