#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from PIL import Image

import os
import sys, getopt

from os.path import isfile
import numpy as np

import ImageUtils
from pybrain.tools.xml.networkreader import NetworkReader


# image parameters
# chunk width and height calculated to split a 1280x720 image in a 20x20 grid
chunkWidth = 64
chunkHeight = 36
chunkSize = chunkWidth * chunkHeight

# filesystem parameters
basePath = os.path.dirname(os.path.realpath(__file__))
chunkDifferencesPath = basePath + '/chunks_differences'
outputPath = basePath + '/output'

def getImageAndMask(imgPath):
    maskPath = imgPath.replace('.jpg','.png')
    if not isfile(imgPath):
        raise IOError("Image %s not found." % (imgPath))

    if not isfile(maskPath):
        raise IOError("Mask %s not found." % (maskPath))

    srcImage = Image.open(imgPath)
    maskImage = Image.open(maskPath)
    return srcImage, maskImage


def getImageAndMaskDifferenceChunks(imgPath):
    srcImage, maskImage = getImageAndMask(imgPath)

    srcImage = ImageUtils.preprocessImage(srcImage,240)
    maskImage = ImageUtils.preprocessMask(maskImage, 210)

    imageChunks = ImageUtils.splitImageInChunks(srcImage, chunkWidth, chunkHeight)
    maskChunks = ImageUtils.splitImageInChunks(maskImage, chunkWidth, chunkHeight)

    return ImageUtils.getDifferenceChunks(imageChunks, maskChunks)


def chunkIsRelevant(c, threshold=250):
    stats = ImageStat.Stat(c)

    if stats.mean[0] > threshold:
        print "mean at %s, ignoring chunk" % (stats.mean)
        return False

    return True

def saveChunks(inputDir, outputDir, relevanceThreshold=255):


    print "saveChunks from %s to %s " % (inputDir, outputDir)
    imageIndex = 1
    for fileName in os.listdir(inputDir):
        path = inputDir + '/' + fileName
        f, file_extension = os.path.splitext(path)

        if file_extension == '.png':
            maskImage = ImageUtils.preprocessMask(Image.open(path),210)
            chunks = ImageUtils.splitImageInChunks(maskImage, chunkWidth, chunkHeight)
            for index, c in enumerate(chunks):
                if(chunkIsRelevant(c)):
                    c.save("%s/chunk-%s-%s-%s%s" % (outputDir, imageIndex, int(index % 20), int(index / 20), file_extension))
            imageIndex+=1
            continue

        if not isfile(path) or file_extension != '.jpg':
            continue


        print "- opening "+path

        srcImage = ImageUtils.preprocessImage(Image.open(path), 240)

        if file_extension == '.jpg':
            chunks = ImageUtils.splitImageInChunks(srcImage, chunkWidth, chunkHeight)

        for index, c in enumerate(chunks):
            savePath = "%s/chunk-%s-%s-%s%s" % (outputDir, imageIndex, int(index % 20), int(index / 20), file_extension)
            if(chunkIsRelevant(c, relevanceThreshold)):
                c.save(savePath, quality=95)

    print "savechunks - finished"



def saveDifferenceChunks(inputDir, outputDir, alsoSaveChunks=False, relevanceThreshold=255):
    print "saveDifferenceChunks from %s to %s " % (inputDir, outputDir)
    if alsoSaveChunks:
        saveChunks(inputDir, inputDir+ "/chunks")

    imageIndex = 1;
    for fileName in os.listdir(inputDir):
        path = inputDir + '/' + fileName
        f, file_extension = os.path.splitext(path)
        if not isfile(path) or file_extension != '.jpg':
            continue

        # print "- opening "+path
        chunks = getImageAndMaskDifferenceChunks(path)
        for index, c in enumerate(chunks):
            savePath = "%s/chunk-%s-%s-%s%s" % (outputDir, imageIndex, int(index % 20), int(index / 20), file_extension)
            print "chunk "+savePath
            if(chunkIsRelevant(c, relevanceThreshold)):
                c.save(savePath, quality=95)
        imageIndex+=1
    print "saveDifferenceChunks - finished"



# NETWORK STUFF

def activateOnImage(fnn, layerpath, saveWrongChunks=False, breakOnError=True, saveWrongImages=False):
    try:
        chunks = getImageAndMaskDifferenceChunks(layerpath)
    except IOError as e:
        print e
        return
    except ValueError as e:
        print e
        return

    fileName = os.path.basename(layerpath)

    index=1
    needToSaveImage = False
    for c in chunks:
        cFlattened = np.asarray(c).flatten()
        estimate = fnn.activate(cFlattened)
        if estimate[0] < 0.7:
            needToSaveImage = True
            print "estimated error on %s. probability OK / KO: %s/%s" % (fileName, estimate[0], estimate[1])
            if saveWrongChunks:
                c.save(outputPath+"/chunk-%s.jpg" %(index))

            if breakOnError:
                break
            index+=1

    if needToSaveImage and saveWrongImages:
        srcImage, maskImage = getImageAndMask(layerpath)
        srcImage.save(outputPath+"/image-%s-original-image.jpg" %(index))
        maskImage.save(outputPath+"/image-%s-original-mask.png" %(index))

        srcImage = ImageUtils.preprocessImage(srcImage,240)
        maskImage = ImageUtils.preprocessMask(maskImage, 210)
        srcImage.save(outputPath+"/image-%s-elab-image.jpg" %(index))
        maskImage.save(outputPath+"/image-%s-elab-mask.png" %(index))

    return


def activateOnFolder(fnn, folder):
    print "activating on %s" % (folder)
    if not os.path.isdir(folder):
        print "folder %s not found" % (folder)
        return

    for content in os.listdir(folder):
        path =folder+"/"+content
        if os.path.isfile(path) and -1 != content.find(".jpg"):
            activateOnImage(fnn, path,False,True, True)
        elif os.path.isdir(path):
            activateOnFolder(fnn, path)
    return



try:
  opts, args = getopt.getopt(sys.argv[1:],"c:",["check="])
except getopt.GetoptError:
  print 'nope'
  sys.exit(2)

if len(sys.argv) < 2:
    print 'use imasko.py -c|--check <inputfolder>'
    sys.exit(2)

fnn = NetworkReader.readFrom("network.xml")
result_classes = ['Ok','Not Ok']

for opt, arg in opts:
  if opt in ("-c", "--check"):
     folder = basePath+"/"+arg
     if folder.endswith("/"):
        folder = folder[:-1]
     activateOnFolder(fnn, folder)