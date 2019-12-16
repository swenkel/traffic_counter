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
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import pickle
import glob
import matplotlib.pyplot as plt
import seaborn as sns
#                                                                              #
################################################################################


################################################################################
#                                                                              #
# functions and classes                                                        #
def generateFolders(df:pd.DataFrame):
    """
    Create folders to store videos
    """
    def createGraphicsFolders(folderName:str):
        """
        Create subfolders for a specific camera.
        """
        if not os.path.exists("./graphics/"+folderName+"/"):
            try:
                os.makedirs("./graphics/"+folderName+"/")
            except:
                print("Error while creating folder to store detections of "+folderName+".")

    if not os.path.exists("./graphics/"):
        try:
            os.makedirs("./graphics/")
        except:
            print("Error while creating detections folder.")

    for folderName in df["CameraName"]:
        createGraphicsFolders(folderName)


def generateDataFrames(camera:str)->pd.DataFrame:
    """
    Generates a DataFrame of all detections of format:

    | datetime | No. cars | No. trucks | No. Busses |
    """
    def getTimeStamp(filename:str)->np.datetime64:
        """
        Get datetime from filename
        """
        strSplits = filename.split("/")[-1].split(".")[0].split("_")

        datetimeString = strSplits[1]+"-"+strSplits[2]+"-"+strSplits[3]+" "+\
                         strSplits[4]+":"+strSplits[5]+":"+strSplits[5]
        return np.datetime64(datetimeString)


    def extractDetections(detection:str)->np.ndarray:
        """
        Extract detection from each
        """
        results = pickle.load(open(detection, "rb"))
        timeStamp = getTimeStamp(detection)
        detectionsC = {'Cars':0,
                       'Trucks':0,
                       'Busses':0}
        for i in range(len(results[0])):
            for j in range(len(results[0][i])):
                dect = results[0][i][j][-1]
                if int(dect) == 2:
                    detectionsC['Cars'] += 1
                elif int(dect) == 5:
                    detectionsC['Busses'] += 1
                elif int(dect) == 7:
                    detectionsC['Trucks'] += 1
        frames = len(results[0])
        detectionsC['Cars_avg'] = int(np.ceil(detectionsC['Cars']/frames))
        detectionsC['Busses_avg'] = int(np.ceil(detectionsC['Busses']/frames))
        detectionsC['Trucks_avg'] = int(np.ceil(detectionsC['Trucks']/frames))
        return (timeStamp,detectionsC['Cars'],detectionsC['Busses'],detectionsC['Trucks'],detectionsC['Cars_avg'],detectionsC['Busses_avg'],detectionsC['Trucks_avg'])


    dumps = np.sort(glob.glob("./detections/"+camera+"/*.p"))
    detections = Parallel(n_jobs=1, prefer="threads")(
        delayed(extractDetections)(detection)
            for detection in dumps
    )
    detections = np.array(detections)
    cols = ["TimeStamp", "Cars", "Busses", "Trucks", "Cars_avg", "Busses_avg", "Trucks_avg"]
    detectionsDF = pd.DataFrame(detections, columns=cols)
    detectionsDF.to_csv("./detections/"+camera+".csv", index=False)
    return detectionsDF


def createGraphics(camera:str,
                   df:pd.DataFrame):
    plt.figure(figsize=(10,5))
    plt.plot(df["Cars"], label="Cars")
    plt.plot(df["Busses"], label="Busses")
    plt.plot(df["Trucks"], label="Trucks")
    plt.xlabel("Video number/time")
    plt.ylabel("Vehicle detection count != vehicle count")
    plt.legend()
    plt.title(camera)
    plt.savefig("./graphics/"+camera+"_"+"plot.png")
    plt.savefig("./graphics/"+camera+"_"+"plot.pdf")
    plt.close()
    plt.figure(figsize=(10,5))
    plt.plot(df["Cars_avg"], label="Cars")
    plt.plot(df["Busses_avg"], label="Busses")
    plt.plot(df["Trucks_avg"], label="Trucks")
    plt.xlabel("Video number/time")
    plt.ylabel("mean vehicle detection count != vehicle count")
    plt.legend()
    plt.title(camera)
    plt.savefig("./graphics/"+camera+"_"+"plot_avg.png")
    plt.savefig("./graphics/"+camera+"_"+"plot_avg.pdf")
    plt.close()

#                                                                              #
################################################################################




def main():
    print("=" * 80)
    cameraInfo = pd.read_csv("./cameraInfo/camera_info.csv")
    generateFolders(cameraInfo)
    for camera in cameraInfo["CameraName"]:
        print("+--+" * 10)
        print("Extracting and visualizing results of", camera)
        detections = generateDataFrames(camera)
        createGraphics(camera, detections)
        print("+--+" * 10)
    print("="*80)

if __name__ == "__main__":
    main()
