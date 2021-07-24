!cp /content/drive/MyDrive/image_dataset/img.zip /content/
!unzip /content/img.zip -d /content/

# import the necessary packages
from tensorflow.data import AUTOTUNE
from imutils import paths
import tensorflow as tf
import numpy as np
import time
import os

def load_images(imagePath):
	# read the image from disk, decode it, resize it, and scale the
	# pixels intensities to the range [0, 1]
	image = tf.io.read_file(imagePath)
	image = tf.image.decode_jpeg(image, channels=3)
	image = tf.image.resize(image, (256, 256)) / 255.0
                                                                                   #images loaded
	# grab the label and encode it
	label = tf.strings.split(imagePath, os.path.sep)[1]
	oneHot = label == classNames
	encodedLabel = tf.argmax(oneHot)

	# return the image and the integer encoded label
	return (image, encodedLabel)
  
  
  
args = {
    "dataset": "img"
}

# grab the list of images in our dataset directory and grab all
# unique class names
print("[INFO] loading image paths...")
imagePaths = list(paths.list_images(args["dataset"]))                               #path of images loaded
classNames = np.array(sorted(os.listdir(args["dataset"])))

folders = os.listdir(args["dataset"])
classNames = np.array([])
for gender in folders:                                                                  # subclasses of gender added
  subfol = os.listdir('img/' + gender)
  # classNames = np.append(classNames, 
  for dress in subfol:
    classNames = np.append(classNames,tf.strings.join([gender,dress], separator='-'))
  # print(subfol)
  
info = tf.strings.split(imagePaths[0], os.path.sep)

# view=""
# for ch in info[-1]:
#   if((ch<'z' and ch>'a') or (ch<'Z' and ch>'a')):
#     view+= ch                                                                    #modified labels from gender to gender sub classes
gender = info[1]
dress = info[2]
label = tf.strings.join([gender,dress], separator='-')
print(label)

# initialize batch size and number of steps
BS = 8
NUM_STEPS = 6589

# build the dataset and data input pipeline
print("[INFO] creating a tf.data input pipeline..")
dataset = tf.data.Dataset.from_tensor_slices(imagePaths)

options = tf.data.Options()
options.experimental_optimization.noop_elimination = True
options.experimental_optimization.map_vectorization.enabled = True
options.experimental_optimization.autotune_ram_budget= True
options.experimental_optimization.apply_default_optimizations = False
dataset = dataset.with_options(options)

dataset = (dataset                                                                       #creating dataset
    .shuffle(52712)
    .cache()  # Cache data
    .map(  # Reduce memory usage
        load_images,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BS)
    .prefetch(  # Overlap producer and consumer works
        tf.data.AUTOTUNE
    )
    
import matplotlib.pyplot as plt
datasetGen = iter(dataset)            
sample_images,sample_labels = next(iter(dataset))                 
photo = sample_images[5]
plt.imshow(photo)
example_gen_output_y = generator_g(photo[tf.newaxis,...], training=False)
example_disc_out = discriminator_y(photo[tf.newaxis,...], training=False)
print(example_disc_out.shape)

plt.figure(figsize=(10,10))

plt.subplot(1,3,1)
plt.imshow(photo, vmin=0, vmax=255) 

plt.subplot(1,3,2)                                                        #visualisation of dscriminator and generator output
plt.imshow(example_gen_output_y[0,...]) 

plt.subplot(1,3,3)
m = example_disc_out[0,...,-1].numpy()*1000
im = plt.imshow(m, vmin=-20, vmax=20, cmap='RdBu_r')
plt.colorbar(im,fraction=0.046, pad=0.04)

plt.show()

generated_image = generator_g(photo[tf.newaxis,...], training=False)
plt.imshow(generated_image[0, :, :, 0])
decision = discriminator_y(generated_image)
print (decision)                                               
