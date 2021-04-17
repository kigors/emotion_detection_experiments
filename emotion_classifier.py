import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import configparser
from pathlib import Path


class EmotionClassifier:
    def __init__(self, config_path) -> None:
        """
        Load trained model based on config.ini file at config_path location.
        In the same folder with config there should be model directory (checkpoint).
        model folder should contain files in tensorflow saved_model format 
        (model folder name matches with checkpoint value in config.ini).
        Class supports classifier models only (no support for multihead or regression
        models).
        """
        # parse config.ini
        model_dir = Path(config_path).parent
        cfg_file = Path(config_path)
        if not cfg_file.exists():
            raise OSError('Model configuration file not found: ' + str(cfg_file))
        
        config = configparser.ConfigParser()
        
        if config.read(cfg_file) == []:
            raise OSError('Failed to parse model config file: ' + str(cfg_file))
            
        self.name = config['MODEL']['name']
        self.dataset = config['MODEL']['dataset']
        self.chpt = config['MODEL']['checkpoint']
        self.img_size = config['IMAGE'].getint('size')
        self.scale = eval(config['IMAGE']['scale'])
        self.labels = {int(index): label for label, index in config.items('LABELS')}

        # load model
        self.model = tf.keras.models.load_model(model_dir / self.chpt)
        
        # warm-up model on a random sample
        sample = tf.random.uniform((1, self.img_size, self.img_size, 3), 0, 1)
        self.model(sample)
        # self.predict([sample, sample, sample])
        
        
    def __str__(self) -> str:
        return "Model info: name = {}, dataset = {}".format(self.name, self.dataset)


    def _preprocess_images(self, images):
        """
        Resize & scale images, prepare a batch. 
        Assume images have different (inconsistent) sizes.
        """
        if len(images) == 0:
            raise ValueError('Provided list of images is empty.')

        tensors = []

        for image in images:
            # verify inputs
            if len(image.shape) != 3 and image.shape[2] != 3:
                raise ValueError(
                    'One or more images is in wrong format.'
                    "Array's rank should be 3, lenght of last dimention should be 3"
                    f'Provided array has shape {image.shape}'
                    )
            
            tensors.append(
                tf.image.resize(image, (self.img_size, self.img_size)) * self.scale
                )
        return tf.stack(tensors, axis=0) 
    
    
    def predict(self, images):
        """
        Make prediction based on list of images.
        Inputs:
            Images is a list of ndarrays of shape (h, w, ch), where
                ch - RGB channels, values in range [0,255] 
                h, w - height and width (could be different for every image)
        Outputs:
            labels - list of named labels corresponding to input images
                     based on best probability
            probabilities - list of probabilities of selected labels
        """
        predictions = self.model(self._preprocess_images(images))
        classes = tf.argmax(predictions, axis=-1).numpy().tolist()
        
        labels = [self.labels[i] for i in classes]
        probabilities = tf.reduce_max(predictions, axis=-1).numpy().tolist()
        
        return labels, probabilities


if __name__ == '__main__':
    from PIL import Image
    import numpy as np

    model = EmotionClassifier('models/bit-m_ds2/config.ini')
    print(model)
    print(model.labels)
    # filename = sys.argv[1]
    filename = 'dataset/train/happy/3.jpg'
    with Image.open(filename) as image:
        img = np.asarray(image.convert('RGB'))
        print(img.shape)
        batch = model._preprocess_images([img, img, img])
        print(batch.shape)
        print(model.predict([img, img, img]))

