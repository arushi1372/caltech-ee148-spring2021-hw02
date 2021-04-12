import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

def visualize(pred_data, real_data):
    data_path = '../RedLights2011_Medium'
    save_path = 'visualizations/'
    for key, value in pred_data.items():
        im = Image.open(os.path.join(data_path, key))
        # Display the image
        plt.imshow(im)
        # Get the current reference
        ax = plt.gca()
        # Create a Rectangle patch
        for box in value:
            tlx, tly, brx, bry = box[0], box[1], box[2], box[3]
            rect = Rectangle((bry, tlx), (tly - bry), (brx - tlx),
                              linewidth=1, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
        
        real_boxes = real_data[key]
        for box in real_boxes:
            tlx, tly, brx, bry = box[0], box[1], box[2], box[3]
            rect = Rectangle((bry, tlx), (tly - bry), (brx - tlx), 
                              linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)

        plt.savefig(save_path + key)
        plt.close()

with open('hw02_preds/preds_train.json') as f:
  pred_data = json.load(f)
with open('hw02_annotations/annotations_train.json') as f:
  real_data = json.load(f)
# with open('hw02_preds/preds_test.json') as f:
#   pred_data = json.load(f)
# with open('hw02_annotations/annotations_test.json') as f:
#   real_data = json.load(f)
visualize(pred_data, real_data)