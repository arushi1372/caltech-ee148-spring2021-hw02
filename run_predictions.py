import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

def read_template1():
    ''' 
    Read in an example of a red light, selected from image 10.
    '''
    data_path = '../RedLights2011_Medium'
    I = Image.open(os.path.join(data_path,"RL-010.jpg"))
    I = np.asarray(I)
    return I[28:49,317:346]

# def read_template2():
#     ''' 
#     Read in an example of a red light, selected from image 1.
#     '''
#     data_path = '../RedLights2011_Medium'
#     I = Image.open(os.path.join(data_path,"RL-001.jpg"))
#     I = np.asarray(I)
#     return I[153:159,313:323]

def read_template2():
    ''' 
    Read in an example of a red light, selected from image 1.
    '''
    data_path = '../RedLights2011_Medium'
    I = Image.open(os.path.join(data_path,"RL-043.jpg"))
    I = np.asarray(I)
    return I[129:143,190:203]


def normalize(a):
    return a.flatten()/np.linalg.norm(a.flatten())

def compute_convolution(I, T, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    # (n_rows,n_cols,n_channels) = np.shape(I)

    # image = normalize(I)
    # kernel = normalize(T)
    # image_orig = np.reshape(image, (I.shape[0], I.shape[1], I.shape[2]))
    # kernel_orig = np.reshape(kernel, (T.shape[0], T.shape[1], T.shape[2]))
    # kernel_orig = np.flipud(np.fliplr(kernel_orig))    # Flip the kernel
    # output = np.zeros_like(image_orig)            # convolution output
    
    # # Add zero padding to the input image 
    # image_padded = np.zeros((image_orig.shape[0] + (kernel_orig.shape[0]-1), 
    #                          image_orig.shape[1] + (kernel_orig.shape[1]-1),
    #                          image_orig.shape[2] + (kernel_orig.shape[2]-1)))   
    # image_padded[(kernel_orig.shape[0]//2):-(kernel_orig.shape[0]//2), 
    #              (kernel_orig.shape[1]//2):-(kernel_orig.shape[1]//2),
    #              (kernel_orig.shape[2]//2):-(kernel_orig.shape[2]//2)] = image_orig

    # image = image_orig[:,:,0] 
    # kernel = kernel_orig[:,:,0]        
    # for x in range(image.shape[1]):     # Loop over every pixel of the image
    #     for y in range(image.shape[0]):
    #         # element-wise multiplication of the kernel and the image
    #         output[y,x,0]=(kernel*image_padded[y:y + kernel.shape[0], x:x + kernel.shape[1], 0]).sum()
    
    # image = image_orig[:,:,1]   
    # kernel = kernel_orig[:,:,1]      
    # for x in range(image.shape[1]):     # Loop over every pixel of the image
    #     for y in range(image.shape[0]):
    #         # element-wise multiplication of the kernel and the image
    #         output[y,x,1]=(kernel*image_padded[y:y + kernel.shape[0], x:x + kernel.shape[1], 1]).sum()
    
    # image = image_orig[:,:,2] 
    # kernel = kernel_orig[:,:,2]        
    # for x in range(image.shape[1]):     # Loop over every pixel of the image
    #     for y in range(image.shape[0]):
    #         # element-wise multiplication of the kernel and the image
    #         output[y,x,2]=(kernel*image_padded[y:y + kernel.shape[0], x:x + kernel.shape[1], 2]).sum()
    
    # plt.imshow(output[:,:,0], cmap='gray')
    # plt.show()

    # row_padding = (T.shape[0] - 1)/2

    # heatmap = np.random.random((n_rows, n_cols))
    windows = np.lib.stride_tricks.sliding_window_view(I, T.shape)
    heatmap = np.zeros((I.shape[0], I.shape[1]))
    for col_ind, axis1 in enumerate(windows):
        for row_ind, window in enumerate(axis1):
            heatmap[col_ind][row_ind] = (np.inner(normalize(window).flatten(), normalize(T).flatten()))

    plt.imshow(heatmap, cmap='gray')
    plt.show()
    return heatmap


def predict_boxes(heatmap, T):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    threshold = 0.97
    for row_idx, row in enumerate(heatmap):
        for col_idx, col in enumerate(row):
            if col > threshold:
                tl_row = row_idx
                tl_col = col_idx
                br_row = row_idx + T.shape[0]
                br_col = col_idx + T.shape[1]
                output.append([tl_row, tl_col, br_row, br_col, heatmap[row_idx, col_idx]])

    return output


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    T1 = read_template1()
    # T2 = read_template2()
    heatmap = compute_convolution(I, T1)
    # heatmap = compute_convolution(I, T2)
    # helloooo
    output = predict_boxes(heatmap, T)

    for i in range(len(output)):
        assert len(output[i]) == 5
        # assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../RedLights2011_Medium'

# load splits: 
split_path = 'hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = 'hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Make predictions on the training set.
'''
preds_train = {}
file_names_train = ['RL-010.jpg']
for i in range(len(file_names_train)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
