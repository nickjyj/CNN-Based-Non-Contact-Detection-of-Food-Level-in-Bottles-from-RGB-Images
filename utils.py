import numpy as np
import numpy.matlib
from natsort import natsorted
from PIL import Image
from random import shuffle,seed
from os import listdir,mkdir
from os.path import join,isdir,isfile 
import matplotlib.pyplot as plt
from skimage.transform import resize
from collections import defaultdict



def rgb2gray(np_imgs):
    """
    convert a 3D image or a 4D numpy array (channel last) to grayscale
    return a grayscale 4D numpy array
    """
    if np_imgs.shape[0]==0:
        return np_imgs

    gray_imgs = np.dot(np_imgs,[0.299, 0.587, 0.114])
    if len(gray_imgs.shape) < 4:
        gray_imgs = gray_imgs[...,np.newaxis]
    return gray_imgs



def generate_labels(len,num_class):
    assert(len % num_class == 0)

    labels = None
    target_num = int(len / num_class)

    for c in range(num_class):
        tmp = np.ones((target_num,)) * c
        if labels is None:
            labels = tmp
        else:
            labels = np.append(labels,tmp,axis=0)
    return labels



def scale_imarray(array,nbits=8):
    scaled_array = array / (2 ** nbits - 1)
    return scaled_array



def save_imarray(array,folder,labels,predicts,precision=2,map='gray'):

    assert(len(array.shape) == 4)

    is_flir = False
    if array.shape[-1] == 1:
        is_flir = True
        newarr = np.squeeze(array,axis=-1)
    else:
        newarr = array
    
    fig, ax = plt.subplots()
    ax.set_axis_off()
    if is_flir:
        im = ax.imshow(newarr[0],cmap=map)
    else:
        im = ax.imshow(newarr[0])
    for i in range(array.shape[0]):

        im.set_data(newarr[i])
        fname = join(folder,str(i+1) + '.jpeg')

        l1 = labels[i].astype(float)
        p1 = predicts[i].astype(float)
        ax.set_title('label: {}\npredict: {}'.format(np.around(l1,precision), np.around(p1,precision))) 
        fig.savefig(fname)



def im_pading_evenly(imgarray,output_size,mode='constant'):
    '''padding horizontally and vertically evenly with zeros
    imgarray: numpy array of one image
    output_size:[height width]
    Note output size should be bigger than original image:
    '''
    diffx, diffy=output_size[0]-imgarray.shape[0], output_size[1]-imgarray.shape[1]
    assert(diffx>=0 and diffy>=0)

    if diffx%2==0:
        x_after=x_before=diffx//2
    else:
        x_before=diffx//2
        x_after=x_before+1

    if diffy%2==0:
        y_after=y_before=diffy//2
    else:
        y_before=diffy//2
        y_after=y_before+1

    return np.pad(imgarray,[(x_before,x_after),(y_before,y_after),(0,0)],mode=mode)



def load_patches_with_labels(folder, im_size=None, is_background=False, is_gray=False, load_bbox=False, load_liquid_ids=False):
    '''load patches in one folder 
    if load_bbox is true, return [patches, labels, ids, bboxs]
    else return [patches, labels, ids, None]
    '''

    patches=[]
    boxs=None
    liquid_id=None
    liquid_ids=None
    lists=natsorted(listdir(folder))
    id=0
    for i in lists:
        if i=='label.txt':
            id=np.loadtxt(join(folder,i))
            continue
        elif i=='coordinates.txt' and load_bbox:
            boxs=np.loadtxt(join(folder,i))
            continue
        elif load_liquid_ids and i=='liquid-label.txt':
            liquid_id=np.loadtxt(join(folder,i))
            continue
        elif i.find('.png')==-1:
            continue

        imname=join(folder,i)
        img = Image.open(imname)
        imgarray = np.array(img,dtype=np.uint8)

        if im_size is not None:
            imgarray=resize(imgarray, (im_size[0], im_size[1]), anti_aliasing=False, preserve_range=True)
            #plt.imshow(imgarray)
            #plt.show()

        patches.append( np.around(imgarray) )
    patches=np.array(patches,dtype=np.uint8)

    if is_gray:
        patches=rgb2gray(patches)

    # generate ids and labels
    if is_background:
        labels=np.zeros(patches.shape[0])
        ids=np.copy(labels)
    else:
        assert(id!=0)
        if load_liquid_ids==True and liquid_id is not None:
            assert(False)
        ids=id*np.ones(patches.shape[0])
        labels=np.ones(patches.shape[0])
        if liquid_id:
            liquid_ids=liquid_id*np.ones(patches.shape[0])
            

    return patches,labels,ids,boxs,liquid_ids



def load_all_patches(folder,im_size,is_gray=False, load_background=True, binary_class=False, load_liquid_ids=False):

    pattern=folder.split('\\')[-1]

    dict_win={}
    dict_b={}
    lists=natsorted(listdir(folder))
    for f_camv in lists:

        if f_camv.find('result')!=-1:
            continue

        fullf=join(folder,f_camv)
        if not isdir(fullf):
            continue

        p=f_camv.split('-l')
        assert(len(p)>1)


        levels_str=p[-1] 
        level_vec=np.empty((0,))
        for l in levels_str:
            if binary_class:
                if int(l) > 2:
                    tmp_l=1
                else:
                    tmp_l=0
            else:
                tmp_l=int(l)
            level_vec=np.append(level_vec,tmp_l)

        print('pattern: {}, folder: {}, orig_label: {}, label: {}'.format(pattern, f_camv, levels_str, level_vec))


        patches_win=None
        patches_b=None

        # look for each patchf in subf
        sub_lists=natsorted(listdir(fullf))
        for patchf in sub_lists:
            if not isdir(join(fullf,patchf)):
                continue
            print('patch folder: {}'.format(patchf))

            # for window
            bottle_lists=natsorted(listdir(join(fullf,patchf,'window')))

            i=0
            for b in bottle_lists:
                if not isdir(join(fullf,patchf,'window',b)):
                    continue
                level=level_vec[i]
                i+=1

                if patches_win is None:
                    patches_win,_,ids_win,boxs_win,liquid_ids_win=load_patches_with_labels(join(fullf,patchf,'window',b),im_size,load_bbox=False,is_gray=is_gray,load_liquid_ids=load_liquid_ids)
                    levels_win=np.ones(ids_win.shape)*level
                    #print('num win patches: {}'.format(patches_win.shape[0]))
                else:
                    patches,_,ids,boxs,liquid_ids=load_patches_with_labels(join(fullf,patchf,'window',b),im_size,load_bbox=False,is_gray=is_gray,load_liquid_ids=load_liquid_ids)
                
                    if patches.shape[0]!=0:
                        patches_win=np.append(patches_win,patches,axis=0)
                        ids_win=np.append(ids_win,ids,axis=0)
                        if load_liquid_ids:
                            liquid_ids_win=np.append(liquid_ids_win,liquid_ids,axis=0)

                        levels=np.ones(ids.shape)*level
                        levels_win=np.append(levels_win,levels,axis=0)
                        #print('num win patches: {}'.format(patches_win.shape[0]))
                        #boxs_win=np.append(boxs_win,boxs,axis=0)
                    else:
                        print('skip empty folder')
            if load_background:
                # for background
                if patches_b is None:
                    patches_b,labels_b,ids_b,_,_=load_patches_with_labels(join(fullf,patchf,'background'),im_size,is_background=True,is_gray=is_gray)
                else:
                    patches,labels,ids,_,_=load_patches_with_labels(join(fullf,patchf,'background'),im_size,is_background=True,is_gray=is_gray)
                    patches_b=np.append(patches_b,patches,axis=0)
                    labels_b=np.append(labels_b,labels,axis=0)
                    ids_b=np.append(ids_b,ids,axis=0)
    
        # increase dimension
        levels_win=levels_win[...,np.newaxis]
        ids_win=ids_win[...,np.newaxis]
        if load_liquid_ids:
            liquid_ids_win=liquid_ids_win[...,np.newaxis]
            labels_win=np.concatenate((levels_win,ids_win,liquid_ids_win),axis=1)
        else:
            labels_win=np.concatenate((levels_win,ids_win),axis=1)
        dict_win[pattern+'-'+f_camv]=[patches_win,labels_win]

        if load_background:
            labels_b=labels_b[...,np.newaxis]
            ids_b=ids_b[...,np.newaxis]
            levels_b=np.concatenate((labels_b,ids_b),axis=1)
            # add to map
            dict_b[pattern+'-'+f_camv]=[patches_b,levels_b]
    return dict_win,dict_b


def group_dict_by_liquid_ids(dict_in, num_outputs, input_size, num_color_channels):
    default_val=np.empty((0,input_size[0],input_size[1],num_color_channels),dtype=np.uint8)
    default_labels=np.empty((0,num_outputs))
    dict_out=defaultdict(lambda: [default_val.copy(),default_labels.copy(),default_labels.copy()])
    for k in dict_in:
        patches=dict_in[k][0]
        labels=dict_in[k][1][:,:num_outputs]
        liquid_ids=dict_in[k][1][:,2]
        uniq=np.unique(liquid_ids)

        for n in uniq:
            mask=liquid_ids==n

            newk='l'+str(int(n))
            patches_tmp=np.append(dict_out[newk][0],patches[mask],axis=0)
            labels_tmp=np.append(dict_out[newk][1],labels[mask],axis=0)
            liquid_ids_tmp=np.append(dict_out[newk][2],liquid_ids[mask][...,np.newaxis],axis=0)
            dict_out[newk]=[patches_tmp,labels_tmp,liquid_ids_tmp]

    return dict_out
