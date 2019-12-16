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
import datetime
import urllib.request as ur
import wget
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

#                                                                              #
################################################################################


################################################################################
#                                                                              #
# functions and classes                                                        #
def generateFolders(df:pd.DataFrame):
    """
    Create folders to store videos
    """
    def createVideoFolders(folderName:str):
        """
        Create subfolders for a specific camera.
        """
        if not os.path.exists("./videos/"+folderName+"/"):
            try:
                os.makedirs("./videos/"+folderName+"/")
            except:
                print("Error while creating folder to store videos of "+folderName+".")

    if not os.path.exists("./videos/"):
        try:
            os.makedirs("./videos/")
        except:
            print("Error while creating video folder.")

    for folderName in df["CameraName"]:
        createVideoFolders(folderName)

def downloadVideo(df:pd.DataFrame,
                  camera:str):
    """
    Download videos using a poor man's scheduler

    NB!: might run out of sync after a while - avg. download time 1s
    no check if video was really updated
    """
    url = df[df["CameraName"] == camera]["URL"].values[0]
    interval = df[df["CameraName"] == camera]["UpdateInterval"]
    PATH = "./videos/"+camera+"/"
    while True:
        filename = camera+"_"+datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+".mp4"
        print(filename)
        #ur.urlretrieve(url, filename=PATH+filename)
        wget.download(url,out=PATH+filename)
        time.sleep(interval*60-1)

def downloadAllVideos(df:pd.DataFrame):
    """
    Download all videos using threads as a poor man's cronjob
    """
    cameras = df["CameraName"].unique()
    Parallel(n_jobs=len(cameras), prefer="threads")(
        delayed(downloadVideo)(df,camera)
        for camera in cameras
    )

#                                                                              #
################################################################################




def main():
    print("=" * 80)
    cameraInfo = pd.read_csv("./cameraInfo/camera_info.csv")
    print("+--+" * 10)
    print("Generating folders")
    generateFolders(cameraInfo)
    downloadAllVideos(cameraInfo)
    print("+--+" * 10)
    print("="*80)

if __name__ == "__main__":
    main()
