#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# In[2]:


IMAGE_SIZE = [224,224]


# In[3]:


train_path = 'C:/Users/LENOVO/Desktop/data/train'
valid_path = 'C:/Users/LENOVO/Desktop/data/train/val'


# In[4]:


vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[5]:


for layer in vgg.layers:
    layer.trainable = False


# In[6]:


folders = glob('C:/Users/LENOVO/Desktop/data/train/*')


# In[7]:


vgg.layers


# In[8]:


x = Flatten()(vgg.output)
top_model = Dense(len(folders), activation='softmax')(x)


# In[9]:


model = Model(inputs=vgg.input, outputs=top_model)


# In[10]:


model.summary()


# In[11]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[12]:


from keras_preprocessing.image import ImageDataGenerator


# In[13]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:/Users/LENOVO/Desktop/data/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('C:/Users/LENOVO/Desktop/data/val',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[14]:


r=model.fit_generator(training_set,
                         samples_per_epoch = 64,
                         nb_epoch = 5,
                         validation_data = test_set,
                         nb_val_samples = 32)


# In[15]:


plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')


# In[16]:


import tensorflow as tf

from keras.models import load_model

model.save('faceRecog_new_model.h5')


# In[17]:


from keras.models import load_model


# In[18]:


m = load_model('faceRecog_new_model.h5')


# In[19]:


from keras.preprocessing import image


# In[160]:


test_image = image.load_img('C:/Users/LENOVO/Desktop/data/val/images(26).jpg', 
               target_size=(224,224))


# In[161]:


test_image


# In[162]:


test_image = image.img_to_array(test_image)


# In[163]:


test_image.shape


# In[164]:


import numpy as np 


# In[165]:


test_image = np.expand_dims(test_image, axis=0)


# In[166]:


test_image.shape


# In[167]:


test_image


# In[168]:


result = m.predict(test_image)


# In[169]:


result


# In[170]:


if result[0][0] == 1.0:
    print('Ben Afflek')
else:
    print('Elton John')


# In[ ]:




