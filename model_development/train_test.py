import pandas as pd
import os
import numpy as np
import random
import pickle
from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.applications import DenseNet201
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import preprocess_input
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from keras.callbacks import EarlyStopping

RETRAIN = False

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    device = '/GPU:0'
else:
    print("No GPU found, using CPU instead.")
    device = '/CPU:0'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

with tf.device(device):

    def preprocess_image(img):
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = preprocess_input(img)
        return img
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image
    )
    
    prediction_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image
        )

    data = pd.read_csv('processed_data.csv')
    data['image'] = data['image'].apply(lambda x: 'images/' + x + '.jpg')

    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in range(7):
        temp = data[data['label'] == i]
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(temp['image'], temp['label'], test_size=0.2, random_state=random.randint(0, 1000))
        X_train.extend(X_train_temp)
        X_test.extend(X_test_temp)
        y_train.extend(y_train_temp)
        y_test.extend(y_test_temp)
    
    train_df = pd.DataFrame({'filename': X_train, 'class': [str(i) for i in y_train]})
    test_df = pd.DataFrame({'filename': X_test, 'class': [str(i) for i in y_test]})

    image_size = (256, 256)

    train_generator = train_datagen.flow_from_dataframe(train_df, x_col='filename', y_col='class', target_size=image_size, class_mode='categorical', batch_size=16)
    test_generator = test_datagen.flow_from_dataframe(test_df, x_col='filename', y_col='class', target_size=image_size, class_mode='categorical')

    if os.path.isfile('model3.h5'):
        model = load_model('model3.h5')
        
    else:
        base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

        for layer in base_model.layers:
            layer.trainable = False
           
        x = base_model.output
        x = MaxPooling2D()(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(256, (3, 3), activation='relu' , padding="same")(x)
        x = Conv2D(256, (3, 3), activation='relu' , padding="same")(x)
        x = Conv2D(256, (3, 3), activation='relu' , padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)
        x = Dropout(0.3)(x)

        x = Conv2D(128, (3, 3), activation='relu' , padding="same")(x)
        x = Conv2D(128, (3, 3), activation='relu' , padding="same")(x)
        x = Conv2D(128, (3, 3), activation='relu' , padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)
        x = Dropout(0.3)(x)
        
        x = Conv2D(64, (3, 3), activation='relu' ,padding="same")(x)
        x = Conv2D(64, (3, 3), activation='relu' ,padding="same")(x)
        x = Conv2D(64, (3, 3), activation='relu' ,padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)

        x = Flatten()(x)
        x = Dense(512)(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256)(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        
        predictions = Dense(7, activation='softmax')(x)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)

        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(train_generator, epochs=40, batch_size=64, verbose=1, validation_data=test_generator, callbacks=[early_stopping])
        
        with open('history.pkl', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
            
        history = history.history
        
        model.save('model.h5')
        

    if RETRAIN:
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=8)
        
        history = model.fit(train_generator, epochs=15, batch_size=64, verbose=1, validation_data=test_generator, callbacks=[early_stopping])
        
        if os.path.isfile('history.pkl'):
            with open('history.pkl', 'rb') as file_pi:
                old_history = pickle.load(file_pi)
        else:
            old_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
            
        for key in history.history.keys():
            old_history[key].extend(history.history[key])
            
        with open('history.pkl', 'wb') as file_pi:
            pickle.dump(old_history, file_pi)
            
        history = old_history
        
        model.save('model.h5')
        
        # model.summary()
      
    data = pd.read_csv('processed_data.csv')
    data['image'] = data['image'].apply(lambda x: 'images/' + x + '.jpg')
    test_labels = data['label'].tolist()
    prediction_generator = prediction_datagen.flow_from_dataframe(data, x_col='image', target_size=image_size, class_mode=None, shuffle=False)    
    stimulate_predictions =[]
    for _ in range(10):
        predictions = model.predict(prediction_generator, verbose=1)
        stimulate_predictions.append(predictions)
    predictions = np.mean(np.array(stimulate_predictions), axis=0)
    labels = np.argmax(predictions, axis=1)

    temp = pd.DataFrame(predictions, columns=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'])
    temp['Image'] = data['image'].values
    temp['Predicted Class'] = labels
    temp['Actual Label']= test_labels
    df = temp[['Image', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC', 'Predicted Class', 'Actual Label']]
    
    df.to_csv('predictions.csv', index=False)
    
    accuracy = accuracy_score(test_labels, labels)
    print(f'Accuracy: {accuracy}')
    
    df = pd.read_csv('predictions.csv')
     
    fig , ax = plt.subplots(2, 1, figsize=(10, 11))
    success_rate = df[df['Predicted Class'] == df['Actual Label']].groupby('Actual Label').size() / df.groupby('Actual Label').size()
    diseases = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    success_rate.index = success_rate.index.map(lambda x: diseases[int(x)])

    # Use a color palette and add the count above each bar
    sns.barplot(x=success_rate.index, y=success_rate.values, hue=success_rate.index, ax=ax[0], palette='viridis')
    for i, v in enumerate(success_rate.values):
        ax[0].text(i, v + 0.01, str(round(v, 2)), color='black', ha='center')
    ax[0].set_title('Probability of Successful Prediction for Each Label')
    ax[0].set_ylabel('Success Rate')
    ax[0].set_xlabel('Disease')
    
    total_count = df.groupby('Actual Label').size()
    successful_count = df[df['Predicted Class'] == df['Actual Label']].groupby('Actual Label').size()

    count_df = pd.DataFrame({'Total Count': total_count, 'Successful Prediction Count': successful_count}).reset_index()

    melted_df = count_df.melt(id_vars='Actual Label', var_name='Type', value_name='Count')
    melted_df['Actual Label'] = melted_df['Actual Label'].map(lambda x: diseases[int(x)])

    bar_plot = sns.barplot(x='Actual Label', y='Count', hue='Type', data=melted_df, palette='viridis', ax=ax[1])

    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.1f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 10), 
                        textcoords = 'offset points')

    ax[1].set_title('Total Count and Successful Prediction Count for Each Disease')
    ax[1].set_ylabel('Count')
    ax[1].legend(frameon=True, title='Type', title_fontsize='13', loc='upper right')
    ax[1].set_xlabel('Disease')

    plt.tight_layout()
    plt.show()
    # Set seaborn style
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], 'o-', label='Train Accuracy')
    plt.plot(history['val_accuracy'], 'o-', label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend(loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], 'o-', label='Train Loss')
    plt.plot(history['val_loss'], 'o-', label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    plt.show()
    