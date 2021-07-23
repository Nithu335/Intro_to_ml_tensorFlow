import tensorflow as tf
#import tensorflow_datsets as tfds
import tensorflow_hub as hub 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import json
import argparse

from PIL import Image
parser = argparse.ArgumentParser ()
parser.add_argument ('--image_path', default='./test_images/hard-leaved_pocket_orchid.jpg', help = 'Path to image.', type = str)
parser.add_argument('--model', help='Trained Model.', type=str)
parser.add_argument ('--top_k', default = 5, help = 'Top K most likely classes.', type = int)
parser.add_argument ('--category_names' , default = 'label_map.json', help = 'Mapping of categories to real names.', type = str)
commands = parser.parse_args()
image_path = commands.image_path
saved_keras_model_filepath = commands.model
top_k = commands.top_k
classes_json_file = commands.category_names

with open(classes_json_file,'r') as f:
    class_names=json.load(f)


    
loaded_model = tf.keras.models.load_model(saved_keras_model_filepath, custom_objects={'KerasLayer':hub.KerasLayer},compile=False)

def process_image(image):
    image = tf.convert_to_tensor(image)
    image /=255
    image = tf.image.resize(image,(224,224)).numpy()
    return image

def predict(image_path, model , top_k):
    im = Image.open(image_path)
    image=np.asarray(im)
    processed_image = process_image(image)
    final_image = np.expand_dims(processed_image,axis=0)
    ps=model.predict(final_image)
    ps_sorted=np.sort(ps[0])[::-1]
    labels_sorted=np.argsort(ps[0])[::-1]
    return ps_sorted[:top_k],labels_sorted[:top_k]

def main():
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_image=process_image(test_image)
    probs, classes = predict(image_path, loaded_model, top_k)
    classes=classes+1

    flower_names=[]
    for label in classes:
        flower_names.append(class_names[label.astype('str')])
        
    print("The top {} probabilities are {}".format(top_k,probs))
    print("The top {} predicted flower names are {}".format(top_k,flower_names))

     #fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
     #ax1.imshow(processed_image, cmap = plt.cm.binary)
     #ax1.axis('off')
     #ax2.barh(np.arange(top_k), probs)
     #ax2.set_aspect(0.1)
     #ax2.set_yticks(np.arange(top_k))
     #ax2.set_yticklabels(flower_names,size='small')
     #ax2.set_title('Class Probability')
     #ax2.set_xlim(0, 1.1)
     #plt.tight_layout()
     #plt.show()
    
if __name__=="__main__":
    main()
    

