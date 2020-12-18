#!/usr/bin/env python
# coding: utf-8

# ## Transfer Learning VGG 16 and VGG 19 using Keras

# In[ ]:


# Transfer Learning Resnet 50


# In[1]:


'''
get_ipython().system('pip show tensorflow')
'''


# In[2]:


'''
get_ipython().system('pip install tensorflow==2.2.0')
'''

# In[3]:


# import the libraries as shown below
# state of art algorithms
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# In[4]:


# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'Datasets/train'
valid_path = 'Datasets/test'


# In[5]:


# Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights

resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)



# In[6]:


# don't train existing weights
for layer in resnet.layers:
    layer.trainable = False


# In[7]:


# useful for getting number of output classes
folders = glob('Datasets/train/*')


# In[8]:


# our layers - you can add more if you want
x = Flatten()(resnet.output)


# In[9]:


prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=resnet.input, outputs=prediction)


# In[10]:



# view the structure of the model
model.summary()


# In[11]:


# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[12]:


# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[13]:


# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('Datasets/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# In[14]:


test_set = test_datagen.flow_from_directory('Datasets/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[15]:


# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=200,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


# In[16]:


# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[17]:


# save it as a h5 file
from tensorflow.keras.models import load_model
model.save('model1_resnet50.h5')


# In[18]:


y_pred = model.predict(test_set)


# In[19]:


y_pred


# In[20]:


import numpy as np
y_pred = np.argmax(y_pred, axis=1)


# In[21]:


y_pred


# In[22]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# In[23]:


model=load_model('model1_resnet50.h5')


# In[25]:


#img_data


# In[26]:


img=image.load_img('Datasets/Test/lamborghini/11.jpg',target_size=(224,224))


# In[27]:


x=image.img_to_array(img)
x


# In[28]:


x.shape


# In[29]:


x=x/255


# In[30]:


x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
img_data.shape


# In[31]:


img_data


# In[32]:


model.predict(img_data)


# In[33]:


a=np.argmax(model.predict(img_data), axis=1)


# In[34]:


a==1

# https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
# https://www.kaggle.com/dskagglemt/car-image-classification-using-resnet50