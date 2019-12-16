################################################################################
#                                                                              #
#                                                                              #
# (c) Simon Wenkel                                                             #
# released under a 3-clause BSD license (see license file)                     #
#                                                                              #
################################################################################



################################################################################
#                                                                              #
# import libraries                                                             #
import os
import time
globalRunTime = time.time()
import datetime
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import ThirdParty.bbox as bbox
import ThirdParty.util as util
from ThirdParty.bbox import *
from ThirdParty.util import *
from ThirdParty.preprocess import prep_image, inp_to_image, letterbox_image
from ThirdParty.darknet import Darknet
import cv2
import pickle

#                                                                              #
################################################################################


################################################################################
#                                                                              #
# functions and classes                                                        #
def parseARGS():
    """
    Parsing arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="ThirdParty/cfg/yolov3.cfg")
    parser.add_argument("--weights", default="ThirdParty/weights/yolov3.weights")
    parser.add_argument("--res", default="608")
    parser.add_argument("--confidence", default=0.5)
    parser.add_argument("--nms_thresh", default = 0.4)
    args = parser.parse_args()
    config = {}
    config["cfg"] = args.cfg
    config["weights"] = args.weights
    config["res"] = args.res
    config["confidence"] = args.confidence
    config["nms_thresh"] = args.nms_thresh
    return config



def generateFolders(df:pd.DataFrame):
    """
    Create folders to store videos
    """
    def createVideoFolders(folderName:str):
        """
        Create subfolders for a specific camera.
        """
        if not os.path.exists("./detections/"+folderName+"/"):
            try:
                os.makedirs("./detections/"+folderName+"/")
            except:
                print("Error while creating folder to store detections of "+folderName+".")

    if not os.path.exists("./detections/"):
        try:
            os.makedirs("./detections/")
        except:
            print("Error while creating detections folder.")

    for folderName in df["CameraName"]:
        createVideoFolders(folderName)



def detectObects(camera:str,
                 inputDimension:int,
                 config:dict,
                 model,
                 CUDA:bool):
    """
    Run object detection on every video
    """
    videos = np.sort(glob.glob("./videos/"+camera+"/*"))
    for video in videos:
        results = []
        detections = []
        cap = cv2.VideoCapture(video)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                img, orig_im, dim = prep_image(frame, inputDimension)
                im_dim = torch.FloatTensor(dim).repeat(1,2)
                if CUDA:
                    im_dim = im_dim.cuda()
                    img = img.cuda()
                with torch.no_grad():
                    output = model(Variable(img), CUDA)
                output = write_results(output, config["confidence"], num_classes=80, nms = True, nms_conf = config["nms_thresh"])
                detections.append(output.cpu().numpy())
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
        results.append(detections)

        print(video, "finished!")
        pickle.dump(results, open("./detections/"+camera+"/"+video.split("/")[-1].split(".")[-2]+".p", "wb"))
#                                                                              #
################################################################################




def main():
    print("=" * 80)
    config = parseARGS()
    cameraInfo = pd.read_csv("./cameraInfo/camera_info.csv")
    generateFolders(cameraInfo)
    print("+" * 80)
    print("Loading model")
    model = Darknet(config["cfg"])
    model.load_weights(config["weights"])
    model.net_info["height"] = config["res"]
    inputDimension = int(config["res"])
    assert inputDimension % 32 == 0
    assert inputDimension > 32
    CUDA = torch.cuda.is_available()
    if CUDA :
        model.cuda()
    model.eval()
    print("+" * 80)
    for camera in cameraInfo["CameraName"]:
        objectRunTimeStart = time.time()
        print("+--+" * 10)
        print("Running object detection on videos from", camera)
        detectObects(camera, inputDimension, config, model, CUDA)
        print("Object detection runtime: {:.2f} min.".format((time.time()-objectRunTimeStart)/60))
        print("+--+" * 10)
    print("Global runtime: {:.2f} min.".format((time.time()-globalRunTime)/60))
    print("="*80)

if __name__ == "__main__":
    main()
