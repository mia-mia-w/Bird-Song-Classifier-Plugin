from waggle import plugin
from waggle.data.audio import Microphone
from tensorflow import keras
import time


def load_model(path_to_model):
    new_model = tensorflow.keras.models.load_model(path_to_model)
    # Evaluate the restored model
    # loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
    # print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
    # print(new_model.predict(test_images).shape)

def main():
    plugin.init()
    mic = Microphone()
    
    # load the machine learning model
    model = load_model('saved_model/my_model')
    
    while True:
        # get data
        sample = mic.record(3)
        sample.save('test.ogg')
        
        # generate prediction
        output = model.predict(sample.data)
        
        # publish predictions
        plugin.publish("bird_song_detected_acafly", 1)
        plugin.publish("bird_song_detected_acowoo", 1)
        plugin.publish("bird_song_detected_aldly", 1)
        
        time.sleep(10)
        
if __name__ = "__main__":
    main()
        
        
        

    

