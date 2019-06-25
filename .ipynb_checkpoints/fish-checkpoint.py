#!/usr/bin/env python
# coding: utf-8

# In[11]:



import os
import sys
import glob
import argparse

import keras
import numpy as np

from keras.models  import Model
from keras.layers import  Input,Dense
from keras.callbacks import Callback
from keras.applications.inception_v3  import InceptionV3
train_dir ='../kagfish/train'


# In[12]:


from keras.preprocessing.image import ImageDataGenerator


# In[13]:


import wandb
from wandb.keras import WandbCallback
wandb.init()
config = wandb.config
config.img_width = 299
config.img_height = 299
config.epochs = 10
config.batch_size = 32


# In[14]:


data_generator = ImageDataGenerator(rescale = 1./255,horizontal_flip=True,validation_split=0.2)

train_generator =data_generator.flow_from_directory("../kagfish/train",target_size=(config.img_width,config.img_height),
                                           batch_size=config.batch_size ,class_mode='categorical',subset="training")
val_generator =data_generator.flow_from_directory("../kagfish/train",target_size=(config.img_width,config.img_height),
                                           batch_size=config.batch_size ,class_mode='categorical',subset="validation")


# In[15]:


nb_train_samples=3025
nb_val_samples = 752


# In[ ]:


base =InceptionV3(weights ='imagenet',include_top=False)


# In[ ]:


from keras.models import Sequential
model = Sequential()
model.add(base)
model.add(keras.layers.GlobalAveragePooling2D(data_format=None))


# In[ ]:


model.add(Dense(8, activation='softmax'))

model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])


# In[ ]:


model.fit_generator(
    train_generator,
    epochs=config.epochs,
    steps_per_epoch = nb_train_samples // config.batch_size,
    validation_data=val_generator,
    validation_steps = nb_val_samples // config.batch_size,
    callbacks=[WandbCallback()])


# In[ ]:





# In[ ]:




