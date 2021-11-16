#from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow.keras.applications import *
import numpy as np
from tqdm import tqdm
import os
import argparse

class DeepModel:
    def __init__(self,model_name,depth):
        self.model = None
        self.__model_name = model_name
        if(not os.path.exists(f"{model_name}")):
            os.mkdir(f"{model_name}")
        if(model_name.lower() == "resnet50"):
            self.model = ResNet50V2()
        elif(model_name.lower() == "resnet101"):
            self.model = ResNet101V2()
        elif(model_name.lower() == "vgg16"):
            self.model = VGG16()
        elif(model_name.lower() == "vgg19"):
            self.model = VGG19()
        elif(model_name.lower() == "inceptionv3"):
            self.model = InceptionV3()
        elif(model_name.lower() == "densenet"):
            self.model = DenseNet121()
        self.__depth = int(depth)
        self.__model_input_shape = self.model.input_shape[1:]
        self.__resize_func = lambda path: cv2.resize(cv2.imread(path),self.__model_input_shape[:2]).reshape((1,*self.__model_input_shape))
    

    def generate_fmap(self,img_path):
        img = self.__resize_func(img_path)
        new_md = tf.keras.models.Model(inputs = self.model.inputs , outputs=  [ o.output for o in self.model.layers if "conv" in o.name])
        outs = new_md.predict(img)
        for i in tqdm(range(self.__depth),desc="Saving feature map ",ncols=100):
            t = outs[i][0]
            sq = int(np.sqrt(t.shape[-1]))
            plt.figure(figsize=(20,20))
            for j in range(sq*sq):
                plt.subplot(sq,sq,j+1)
                plt.imshow(t[:,:,j])
                plt.axis("off")
            output_layername = new_md.outputs[i].name.replace("/","_").replace(":","_")
            plt.savefig(f"./{self.__model_name}/{output_layername}.png")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="give model name and depth")
    parser.add_argument("--model","-m",default = "resnet50")
    parser.add_argument("--depth","-d",default = 2)
    parser.add_argument("--image","-i")
    args = parser.parse_args()

    model = DeepModel(args.model,args.depth)
    model.generate_fmap(args.image)
