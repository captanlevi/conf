import pandas as pd
import numpy as np
import json
import os
import random
from typing import Iterator, Union, List, Dict, Any
import re


class DataManager:
    # video , live video , conference , gameplay , gaming download, streaming
    mainTypes = [
        "Video",
        "Live Video",
        "Conferencing",
        "Gameplay",
        "Gaming Download",
        "Large Download"]
    providerDict = {"Video": ["Netflix", "YouTube", "AmazonPrime", "Disney", "Stan"],
                    "Live Video": ["Twitch", "Seven Live"],
                    "Conferencing": ["Microsoft Teams", "Zoom", "Discord", "SkypeCall", "WhatsAppVoice",
                                     "GoogleMeet", "Webex", "HouseParty", "GoogleHangout", "SkypeForBusiness",
                                     "Facetime"],

                    "Gameplay": ["Destiny 2", "League of Legends", "CoD: Modern Warfare", "World of Warcraft",
                                 "CS:GO", "CoD: Black Ops Cold War", "RedDeadOnline", "Overwatch", "Hearthstone", "Fortnite",
                                 "Genshin Impact", "Escape From Tarkov", "Rocket League", "Among Us", "Battlefront II", "Valorant",
                                 "World of Tanks", "Starcraft 2", "Titanfall 2", "Battlefield V", "Age Of Empires II"],


                    "Gaming Download": ["Steam", "XboxLive", "Epic Games", "DCS World", "Origin", "Riot Games",
                                        "Playstation", "Oculus", "Blizzard", "WarGaming", "2K Sports", "Ubisoft"],


                    "Large Download": ["Discord", "Office365", "Windows Update", "Microsoft", "Apple", "Adobe",
                                       "Apple iOSAppStore", "Xilinx", "CDN-Apple", "Ubuntu", "Spotify", "Symantec"],
                    "Large Upload": []

                    }

    # Features list
    # the number of channels (up,down,bytes,packets and packetLength)
    numNNFeatures = 6
    nnFeatures = ["Array"]

    # bands and thresholds
    defaultThresholds = [64, 1250]

    # Bytes Div factor
    bytesDivFactor = 8

    # Limiting Data so that the machine does not crash
    maxDataPoints = 17000

    # indices
    arrayIndex = dict(
        u_p=2,
        d_p=3,
        u_b=0,
        d_b=1,
        u_pl=4,
        d_pl=5
    )

    @staticmethod
    def get_correct_provider(Type: str, provider: str) -> str:
        gaming_download_corrections = {
            "steam": "Steam",
            "xboxlive": "XboxLive",
            "playstation": "Playstation",
            "epic games": "Epic Games",
            "riot games": "Riot Games",
            "blizzard": "Blizzard"
        }

        if Type == "Gaming Download":
            provider_lowercase = provider.lower()
            return gaming_download_corrections.get(provider_lowercase, provider)

        return provider

    @staticmethod
    def cnnFeaturesNormalizer(features):
        # expects a numpy array of shape (BS, len(cnnFeatures) , 3(high mid low) , Time steps)
        # The catch is the sequence must be the same as the one mentioned in
        # the cnnFeatures list

        features = features.astype(np.float64)
        # first normalizing the up-down-bytes-packets , bu dividing by the max
        # in each row
        features[:, 0:4, :, :] = features[:, 0:4, :, :] / \
            (features[:, 0:4, :, :].max(axis=-1, keepdims=True) + 1e-6)
        # no need to normalize the packetsLengths (already normalized !!!)

        #features[features == 0] = -1
        return features.astype(np.float32)

    # re patterns
    #typeReCompiled = re.compile(r".*\"classification\":{\"type\":\"(?P<name>[a-zA-Z ]*)\"")
    #providerReCompiled = re.compile(r".*\"classification\":{.*\"provider\":\"(?P<name>[a-zA-Z:0-9_\-./ ]*)\"")

    #typeReCompiled = re.compile(r".*\"classification\":{\"type\":\"(?P<name>[^\"]*)\"")
    #providerReCompiled = re.compile(r".*\"classification\":{.*\"provider\":\"(?P<name>[^\"]*)\"")

    typeReCompiled = re.compile(
        r".*\"classification\"\s*:\s*{\s*\"type\"\s*:\s*\"(?P<name>[^\"]*)\"")
    providerReCompiled = re.compile(
        r".*\"classification\"\s*:\s*{.*\"provider\"\s*:\s*\"(?P<name>[^\"]*)\"")
    protoReCompiled = re.compile(
        r".*\"fiveTuple\"\s*:\s*{.*\"proto\"\s*:\s*(?P<name>[0-9]*)\s*[},]")
    threshReCompiled = re.compile(
        r".*\"thresholds\"\s*:\s*(?P<name>[^\"]*)\s*,")

    @staticmethod
    def printSingleRow(row):
        index = DataManager.arrayIndex
        print("Type = {} , provider = {}".format(row.type, row.provider))

        print(
            "sni = {}\ndns = {}\nclassifier = {}".format(
                row.sni,
                row.dns,
                row.classifier))
        print("Interval = {}".format(row.Interval))

        print("*" * 50)
        print("")
        print("down-Bytes-Array\n{}".format(
            (row.Array[index["d_b"]] * DataManager.bytesDivFactor).astype(np.int64)))

        print("*" * 50)
        print("")
        print("up-Bytes-Array\n{}".format(
            (row.Array[index["u_b"]] * DataManager.bytesDivFactor).astype(np.int64)))

        print("*" * 50)
        print("")
        print(
            "down-Packets-Array\n{}".format(row.Array[index["d_p"]].astype(np.int64)))

        print("*" * 50)
        print("")
        print(
            "up-packets-Array\n{}".format(row.Array[index["u_p"]].astype(np.int64)))

        print("*" * 50)
        print("")
        print("down-Packets-Length-Array\n{}".format(row.Array[index["d_pl"]]))

        print("*" * 50)
        print("")
        print("up-packets-Length-Array\n{}".format(row.Array[index["u_pl"]]))

        done = ["Array", "type", "provider", "classifier",
                "sni", "dns", "Interval"]

        ignore = ["sum_data"]
        for key, value in row.iteritems():
            if(key in ignore):
                continue

            if(key not in done):
                print("{} = {}".format(key, value))

    def __init__(self, dataSource: Union[str, List[Dict]], chopSeqLen=30,
                 maxFileRead=None, window=False, sample=False, splits=None):
        """
        splits is a list of dicts ex [ {"Video" : ["Netflix", "Prime"]} , {"Live Video" : [], "Gameplay" : []}, .....]
        advised to get it from splitTree itself
        """

        self.interval = None
        self.df = pd.DataFrame()
        self.chopSeqLen = chopSeqLen
        self.sample = sample
        self.window = window
        self.splits = splits
        if(self.splits is not None):
            self.TPFilter = DataManager.getTPFilter(self.splits)
        else:
            self.TPFilter = DataManager.providerDict
        self.maxFileRead = maxFileRead

        self.numBands = None

        if(self.TPFilter is None):
            self.TPFilter = DataManager.providerDict

        if(self.sample == True):
            assert isinstance(
                dataSource, str) == True, "cant sample data if it is being read directely"
            assert self.splits is not None, "Must provide splits to sample accordingly, if sample flag is True"

        processedForDf = []

        if(isinstance(dataSource, str)):
            dataFiles = os.listdir(dataSource)
            dataFiles = sorted(filter(lambda x: x.endswith(".log"), dataFiles))
            if(self.maxFileRead is not None and self.maxFileRead < len(dataFiles)):
                #dataFiles = random.sample(dataFiles, self.maxFileRead)
                dataFiles = dataFiles[:self.maxFileRead]
                # print("reading {} files".format(len(dataFiles)))

            if(self.sample == True):
                self.probasDct = self.getProbablityDict(dataSource, dataFiles)

            for dataFile in dataFiles:
                dataPath = os.path.join(dataSource, dataFile)
                if(dataPath.endswith(".log") == False):
                    continue
                processedForDf.extend(self.extractDataNew(dataPath))

            self.df = pd.DataFrame(processedForDf)
            self._processRawData()
            # self._printDataSummary()

        elif(isinstance(dataSource, pd.DataFrame)):
            self.df = dataSource
            sample = self.df.sample(1).iloc[0]
            self.interval = sample.Interval
            self.numBands = sample.Array.shape[1]
            self._processRawData()
            # self._printDataSummary()

        else:
            processedForDf.extend(self.extractDataNew(dataSource))
            self.df = pd.DataFrame(processedForDf)

    @staticmethod
    def getTPFilter(splits: List):
        """
        splits is a list of dicts ex [ {"Video" : ["Netflix", "Prime"]} , "Live Video" : [], .....]
        """

        TPFilter = dict()

        for split in splits:
            for tp, providers in split.items():
                if(tp not in TPFilter):
                    TPFilter[tp] = []
                TPFilter[tp].extend(providers)

        return TPFilter

    def normalizePacketLengthArray(self, array, thresh):
        array = np.array(array.tolist())
        thresh = np.array(thresh.tolist())
        array[:, 5, :, :] /= thresh
        array[:, 4, :, :] /= thresh
        retArr = []

        for arr in array:
            retArr.append(arr)

        return retArr

    def __len__(self):
        return len(self.df)

    @staticmethod
    def getTypeFromString(string):
        tp = re.match(DataManager.typeReCompiled, string).group("name")
        return tp

    @staticmethod
    def getProviderFromString(string):
        """
        Change all the gampplay providers names
        """
        p = re.match(DataManager.providerReCompiled, string).group("name")
        if("Steam" in p):
            p = "Steam"

        return p

    @staticmethod
    def getProtoFromString(string):
        proto = re.match(DataManager.protoReCompiled, string).group("name")
        proto = int(proto)
        return proto

    @staticmethod
    def getThreshFromString(string):
        thresh = re.match(DataManager.threshReCompiled, string).group("name")
        thresh = thresh[1:-1].split(",")
        thresh = [int(x) for x in thresh]
        return thresh

    def probeRun(self, rootDirPath, dataFiles):
        """
        The probe run will return a dict of format {Type : {total_count : count , provider_counts : {provider1_count : 23...}}...}
        """
        countDct = dict()
        dirs = dataFiles

        for dir in dirs:
            dataPath = os.path.join(rootDirPath, dir)
            if(dataPath.endswith(".log") == False):
                continue

            print("Starting probe run!!!")
            with open(dataPath, "r") as f:
                for l in f:
                    tp = DataManager.getTypeFromString(l)
                    provider = DataManager.get_correct_provider(
                        tp, DataManager.getProviderFromString(l))

                    if(tp not in self.TPFilter):
                        continue

                    if(len(self.TPFilter[tp]) != 0 and provider not in self.TPFilter[tp]):
                        continue

                    if(tp not in countDct):
                        countDct[tp] = {"totalCount": 0, "providerCounts": {}}

                    if(provider not in countDct[tp]["providerCounts"]):
                        countDct[tp]["providerCounts"][provider] = 0

                    countDct[tp]["totalCount"] += 1
                    countDct[tp]["providerCounts"][provider] += 1

        return countDct

    def getProbablityDict(self, rootDirPath, dataFiles):
        """
        returns probDct which is mapping of (type, provider) tuple to its probablity to get picked
        """
        def getSplitCount(split, countDct):
            count = 0
            for tp, providers in split.items():
                if(tp not in countDct):
                    print("Type {} not found in the dataset".format(tp))
                    continue
                if(len(providers) == 0):
                    count += countDct[tp]["totalCount"]
                else:
                    for provider in providers:
                        count += countDct[tp]["providerCounts"].get(
                            provider, 0)
            return count

        def getSplitCountArray(split, countDct, minCount):
            tp_prov_count_array = []
            for tp, split_providers in split.items():
                if(tp not in countDct):
                    # if the type in split is not in dataset
                    continue

                if(len(split_providers) == 0):
                    providers = list(countDct[tp]["providerCounts"].keys())
                else:
                    # in cases where providers are splecified ie video provider
                    # classification splits
                    providers = split_providers

                for provider in providers:
                    if(provider not in countDct[tp]["providerCounts"]):
                        print(
                            "provider {} mentioned in splits not found in dataset".format(provider))
                        continue
                    count = countDct[tp]["providerCounts"][provider]
                    tp_prov_count_array.append((tp, provider, count))

            tp_prov_count_array.sort(key=lambda x: x[2])

            ret_array = []
            # target for each tp_prov pair to reach
            target_count = minCount / len(tp_prov_count_array)
            for index, item in enumerate(tp_prov_count_array):
                tp, provider, count = item

                if(count < target_count):
                    # will pick all the examples from here, and distribute the
                    # rest for the remaining tp_prov pairs
                    ret_array.append((tp, provider, count))
                    minCount -= count
                    target_count = minCount / \
                        (len(tp_prov_count_array) - 1 - index)

                else:
                    ret_array.append((tp, provider, target_count))

            return ret_array

        countDct = self.probeRun(rootDirPath=rootDirPath, dataFiles=dataFiles)

        print(countDct)
        splitCounts = []  # array to store the found counts of each split
        for split in self.splits:
            splitCounts.append(getSplitCount(split, countDct))

        # now we want to have equal number of counts in each split, equal to
        # the min split count
        minCount = min(min(splitCounts), DataManager.maxDataPoints)

        # now computing number to sample for each type,provider tuple and by
        # that its prob to get picked
        probDct = dict()

        for split in self.splits:
            # need to compute how to distribute the minCount within the tps
            tp_prov_count_array = getSplitCountArray(split, countDct, minCount)

            for item in tp_prov_count_array:
                tp, provider, count = item
                probDct[(tp, provider)] = count / \
                    (countDct[tp]["providerCounts"][provider])

        print("minCount = {}".format(minCount))
        print(probDct)
        return probDct

    def extractDataNew(self, dataPath: Union[List[Dict], str]):
        """
        Extracting data for the new 100 mil sec data

        this function works for extracting data from a file or directely though a list of dicts
        (they have the same format !!!)
        """
        totalData = []
        isDirect = True
        if(isinstance(dataPath, str)):
            isDirect = False

        def makeNpArr(arr):
            arr = np.array(arr)
            return arr.T.astype(np.float64)

        def buildDct(dct):
            dct = dct.copy()
            if(not isDirect):
                rawData = dct.pop("bucket_data")
                flowInfo = dct.get("FlowInfo", dict())
                if("metadata" in flowInfo):
                    metaData = flowInfo.pop("metadata")
                elif("attributes" in flowInfo):
                    metaData = flowInfo.pop("attributes")
                else:
                    metaData = dict()

                dct["sni"] = metaData.get("sni", "_unknown")
                dct["dns"] = metaData.get("dns", "_unknown")

                cls = flowInfo.pop("classification")
                dct["classifier"] = cls.get("classifier", "_unknown")
                if("sum_data" in dct):
                    dct.pop("sum_data")

                # setting type and provider info
                dct["provider"] = DataManager.get_correct_provider(
                    Type=cls["type"], provider=cls["provider"])
                dct["type"] = cls["type"]
            else:
                rawData = dct

            # making the array
            downPacketsArray = makeNpArr(rawData["downPackets"])
            upPacketsArray = makeNpArr(rawData["upPackets"])
            downBytesArray = makeNpArr(
                rawData["downBytes"]) / DataManager.bytesDivFactor
            upBytesArray = makeNpArr(
                rawData["upBytes"]) / DataManager.bytesDivFactor

            upPacketsLengthArray = (
                upBytesArray / (upPacketsArray + 1e-6)) * DataManager.bytesDivFactor
            downPacketsLengthArray = (
                downBytesArray / (downPacketsArray + 1e-6)) * DataManager.bytesDivFactor

            dct["thresholds"] = dct.get(
                "thresholds", DataManager.defaultThresholds).copy()
            dct["thresholds"].append(1500)
            dct["thresholds"] = np.array(dct["thresholds"]).reshape(-1, 1)
            dct["thresholds"][dct["thresholds"] == 0] = 1

            # normalizing it the first time while reading.
            upPacketsLengthArray /= dct["thresholds"]
            downPacketsLengthArray /= dct["thresholds"]

            dct["Array"] = np.array([upBytesArray,
                                     downBytesArray,
                                     upPacketsArray,
                                     downPacketsArray,
                                     upPacketsLengthArray,
                                     downPacketsLengthArray])
            ###################################################################

            # Setting array information
            dct["seqLen"] = dct["Array"].shape[2]
            dct["numBands"] = dct["Array"].shape[1]

            if(self.numBands is None):
                self.numBands = dct["numBands"]
            else:
                assert self.numBands == dct["numBands"], "cannot have data of varying bands in the same DataManager found {} and {}".format(
                    self.numBands, dct["numBands"])

            if(self.window):
                dctArr = self.processDict(dct, int(self.chopSeqLen / interval))
                for d in dctArr:
                    totalData.append(d)

            else:
                chopAt = int(self.chopSeqLen / interval)
                dct["Array"] = dct["Array"][:, :, :chopAt]
                totalData.append(dct)

        path = dataPath

        if not isDirect:
            print("Reading data from a file")
            with open(path, "r") as f:
                for l in f:
                    tp = DataManager.getTypeFromString(l)
                    provider = DataManager.get_correct_provider(
                        Type=tp, provider=DataManager.getProviderFromString(l))
                    if(tp not in self.TPFilter):
                        continue

                    if(len(self.TPFilter[tp]) != 0 and provider not in self.TPFilter[tp]):
                        continue

                    if(self.sample):
                        # reading type from the string then making a choice to
                        # take it or not
                        probDctKey = (tp, provider)
                        if(probDctKey not in self.probasDct):
                            continue
                        prob = self.probasDct[probDctKey]
                        toss = np.random.random()

                        if(toss > prob):
                            continue

                    dct = json.loads(l)
                    interval = dct["Interval"]
                    if(self.interval is None):
                        self.interval = interval
                    assert self.interval == interval, "Found two different intervals {} and {} while reading data".format(
                        self.interval, interval)
                    if(interval * dct["n_points"] < self.chopSeqLen):
                        continue

                    # classifier = dct["FlowInfo"]["classification"]["classifier"]
                    # if classifier.startswith("TPE."):
                    #     continue

                    buildDct(dct)
        else:
            # reading directely
            # print("Reading data directely!!!")
            for dct in dataPath:
                interval = dct["Interval"]
                if(self.interval is None):
                    self.interval = interval
                if(interval * dct["n_points"] < self.chopSeqLen):
                    assert False, "Not enough data , ie chopSeqlen not cleared expected at least {} secs of data found {}".format(
                        self.chopSeqLen, interval * dct["n_points"])
                buildDct(dct)

        return totalData

    def processDict(self, dataDict, seqLen, stride=None):
        """
        not correcting start and end time
        """
        retArr = []
        timeSteps = dataDict["Array"].shape[2]
        if(timeSteps <= seqLen):
            return [dataDict]

        if(stride is None):
            stride = seqLen // 2

        startIndex = 0
        flag = True

        while flag:
            if(startIndex > timeSteps - seqLen):
                flag = False
                # Making seq exactely the same size
                break

            tempDct = dict()

            for key, value in dataDict.items():
                if(key == "Array"):
                    continue
                tempDct[key] = value

            tempDct["Array"] = dataDict["Array"][:,
                                                 :, startIndex: startIndex + seqLen]
            retArr.append(tempDct)
            startIndex += stride

        return retArr

    def _processRawData(self):
        # filtering the cols that we want
        # print(len(self.df))
        nonZeroMask = ((self.df.Array.apply(lambda x: x[DataManager.arrayIndex["u_p"]].sum() > 0)).values & (
            self.df.Array.apply(lambda x: x[DataManager.arrayIndex["d_p"]].sum() > 0)).values)
        self.df = self.df.iloc[nonZeroMask]
        # print(len(self.df))

        # new creating split_key used to make balanced train test and val
        # splits
        self.df["splitKey"] = self.df["type"] + "_" + self.df["provider"]

    @staticmethod
    def correctPacketsLength(arr):
        """
        After any transformation such as band reduction or upsampling we must correct packetlength array
        And normalize it
        """
        index = DataManager.arrayIndex
        arr[index["u_pl"],
            :,
            :] = (arr[index["u_b"],
                      :,
                      :] / (arr[index["u_p"],
                                :,
                                :] + 1e-6)) * DataManager.bytesDivFactor  # upBytes/upPackets
        arr[index["d_pl"],
            :,
            :] = (arr[index["d_b"],
                      :,
                      :] / (arr[index["d_p"],
                                :,
                                :] + 1e-6)) * DataManager.bytesDivFactor  # downBytes/downPackets
        return arr

    @classmethod
    def upSampleArray(cls, arr, factor):
        """
        arr is of shape (6,bands,X)
        where X is divisible by factor
        """
        shape = arr.shape

        cutShape = None
        for sp in range(shape[2], 0, -1):
            if(sp % factor == 0):
                cutShape = sp
                break

        if(cutShape is None):
            assert False, "Upsampling factor is too big , for the current data"

        arr = arr[:, :, :cutShape]
        shape = arr.shape

        assert shape[2] % factor == 0, "Upsampling dim must be divisible by the factor , got {} and {}".format(
            shape[2], factor)

        newDim = shape[2] // factor
        newArr = np.zeros((shape[0], shape[1], newDim), dtype=np.float64)

        for i in range(factor):
            newArr += arr[:, :, i:shape[2]:factor]

        # NOw the packet length array has been wrongly upSampled, correcting it
        newArr = DataManager.correctPacketsLength(newArr)
        return newArr

    def upSampleData(self, factor):
        assert factor > 1, "Must up sample , cant downsample"
        print("Upsampling by factor {}".format(factor))
        self.df["Array"] = self.df["Array"].apply(
            lambda x: DataManager.upSampleArray(x, factor))

        # now applying normalization
        self.df["Array"] = self.normalizePacketLengthArray(
            self.df.Array, self.df.thresholds)
        self.interval = self.interval * factor

    @staticmethod
    def reduceBandArray(array, numReduce):
        """
            arr is of shape (6,bands,X)
            where X is divisible by factor
        """
        arrayBands = array.shape[1]

        assert numReduce > 0
        assert arrayBands > numReduce, "cant reduce {} bands as array only has {} bands".format(
            numReduce, arrayBands)
        newArrayBands = arrayBands - numReduce
        newArr = np.zeros(
            (array.shape[0],
             newArrayBands,
             array.shape[2]),
            dtype=np.float64)

        newArr[:, 0, :] = array[:, 0:numReduce + 1, :].sum(axis=1)
        newArr[:, 1:, :] = array[:, numReduce + 1:, :]
        newArr = DataManager.correctPacketsLength(newArr)
        return newArr

    def reduceBands(self, numReduce):
        assert numReduce > 0
        assert self.numBands > numReduce, "The dataManager only has {} bands cannot reduce {}".format(
            self.numBands, numReduce)

        self.df["Array"] = self.df["Array"].apply(
            lambda x: DataManager.reduceBandArray(x, numReduce))

        # correcting thresholds
        self.df["thresholds"] = self.df["thresholds"].apply(
            lambda x: x[numReduce:])

        # now applying normalization
        self.df["Array"] = self.normalizePacketLengthArray(
            self.df.Array, self.df.thresholds)
        self.numBands = self.numBands - numReduce

    def fftRatio(self, arr):
        F = np.fft.fft(arr) / len(arr)
        F = abs(F[range(int(len(arr) / 2))])[1:]
        #ex = np.exp(F)
        #F = ex/ex.sum()
        F = F / F.sum()

        counts = np.arange(1, len(F) + 1)
        F = F * counts

        return F.sum() / counts.sum()

    @staticmethod
    def makeDfSplittable(df, splitKey="splitKey"):
        series = df[splitKey].value_counts() == 1
        delKeys = []
        for key, condition in zip(series.index, series):
            if(condition == True):
                delKeys.append(key)

        return df[~df[splitKey].isin(delKeys)]

    def getFeatures(self):
        self._addFeatures()

    def _printDataSummary(self):
        print(self.df.type.value_counts())

    def _calcNonZeroMean(self, mainIndex, lookUpKey, bandNumber):
        """
        Want to calc this only for pl
        """
        arrayTp = lookUpKey.split("_")[1]
        if(arrayTp != "pl"):
            return

        def getMean(arr):
            brr = arr[arr > 0]
            if(len(brr) == 0):
                return 0
            return brr.mean()

        self.df["f_nzm" + "-" + lookUpKey + str(bandNumber)] = self.df["Array"].apply(
            lambda x: getMean(x[mainIndex, bandNumber, :]))

    def _calcSparcity(self, mainIndex, lookUpKey, bandNumber):
        """
            has to be calculated only once per direction per band , cause its gonna be the same
        """

        arrayType = lookUpKey.split("_")[1]
        direction = lookUpKey.split("_")[0]

        if(arrayType != "p"):
            return

        self.df["f_sparcity" + "-" + direction + str(bandNumber)] = self.df["Array"].apply(
            lambda x: (x[mainIndex, bandNumber, :] == 0).sum() / x.shape[2])

    def _calcFraction(self, mainIndex, lookUpKey, bandNumber):
        if(lookUpKey.split("_")[1] == "pl"):
            # not gonna do this for packet length
            return
        self.df["f_fraction" + "-" + lookUpKey + str(bandNumber)] = self.df["Array"].apply(
            lambda x: (x[mainIndex, bandNumber, :].sum()) / (x[mainIndex].sum() + 1e-6))

    def _calcChangeCount(self, mainIndex, lookUpKey, bandNumber):
        """
        has to be calculated only once per direction per band , cause its gonna be the same
        """
        arrayType = lookUpKey.split("_")[1]
        direction = lookUpKey.split("_")[0]

        if(arrayType != "p"):
            return

        def changeCount(arr):
            count = 1

            state = 1 if arr[0] > 0 else 0

            for i in range(1, len(arr)):
                if(state == 1 and arr[i] > 0):
                    continue
                elif(state == 1 and arr[i] <= 0):
                    count += 1
                    state = 0
                elif(state == 0 and arr[i] <= 0):
                    continue

                else:
                    count += 1
                    state = 1

            return count / len(arr)

        self.df["f_changeCount" + "-" + direction + str(bandNumber)] = self.df["Array"].apply(
            lambda x: changeCount(x[mainIndex, bandNumber, :]))

    def _calcNonZeroVarience(self, mainIndex, lookUpKey, bandNumber):
        def calcVar(arr):
            brr = arr[arr != 0]
            if(len(brr) == 0):
                return 0
            return brr.var() / (((brr.max() - brr.min()) / 2)**2 + 1e-5)

        self.df["f_nzv" + "-" + lookUpKey + str(bandNumber)] = self.df["Array"].apply(
            lambda x: calcVar(x[mainIndex, bandNumber, :]))

    def _addFeatures(self):
        directions = ["u", "d"]
        arrayTypes = ["b", "p", "pl"]

        for direction in directions:
            for arrayType in arrayTypes:
                lookUpKey = direction + "_" + arrayType
                mainIndex = DataManager.arrayIndex[lookUpKey]
                # overall features goes here
                for band in range(self.numBands):
                    self._calcNonZeroMean(
                        mainIndex=mainIndex,
                        lookUpKey=lookUpKey,
                        bandNumber=band)
                    self._calcSparcity(
                        mainIndex=mainIndex,
                        lookUpKey=lookUpKey,
                        bandNumber=band)
                    self._calcFraction(
                        mainIndex=mainIndex,
                        lookUpKey=lookUpKey,
                        bandNumber=band)
                    self._calcChangeCount(
                        mainIndex=mainIndex,
                        lookUpKey=lookUpKey,
                        bandNumber=band)
                    self._calcNonZeroVarience(
                        mainIndex=mainIndex, lookUpKey=lookUpKey, bandNumber=band)

    @staticmethod
    def dataFramePlotter(df):

        def _bandRatio(direction, subject, bandNumber):
            assert bandNumber <= numBands
            lookUp = direction + "_" + subject

            mainIndex = DataManager.arrayIndex[lookUp]
            tempDf[lookUp + str(bandNumber)] = df["Array"].apply(lambda x: (
                x[mainIndex, bandNumber, :].sum()) / (x[mainIndex].sum() + 1e-6))

        def _nonZeroMean(direction, subject, bandNumber):

            def getMean(arr):
                arr = arr[arr > 0]
                if(len(arr) == 0):
                    return 0
                return arr.mean()

            assert bandNumber <= numBands
            lookUp = direction + "_" + subject

            mainIndex = DataManager.arrayIndex[lookUp]
            tempDf[lookUp + str(bandNumber)] = df["Array"].apply(
                lambda x: getMean(x[mainIndex, bandNumber, :]))

        def _sparcityFunc(direction, subject, bandNumber):
            assert bandNumber <= numBands
            lookUp = direction + "_" + subject
            mainIndex = DataManager.arrayIndex[lookUp]
            tempDf[lookUp + str(bandNumber)] = df["Array"].apply(
                lambda x: (x[mainIndex, bandNumber, :] == 0).sum() / x.shape[2])

        assert len(df) != 0, "cant plot an empty dataFrame"
        assert "provider" in df and "type" in df, "must have type and provider"

        """
        Each bytes and packets array is of shape (N,TS) where N is number of Bands and TS are the num time steps.
        """

        df = df.copy()
        numBands = df.sample(1).iloc[0].Array.shape[1]

        df.provider.value_counts().plot.pie()

        tempDf = pd.DataFrame()
        for band in range(numBands):
            _bandRatio("u", "b", band)
            _bandRatio("d", "b", band)

        tempDf.plot.box(title="Bytes fraction", showfliers=False)

        tempDf = pd.DataFrame()
        for band in range(numBands):
            _bandRatio("u", "p", band)
            _bandRatio("d", "p", band)
        tempDf.plot.box(title="Packets fraction", showfliers=False)

        tempDf = pd.DataFrame()

        for band in range(numBands):
            _nonZeroMean("u", "pl", band)
            _nonZeroMean("d", "pl", band)
        tempDf.plot.box(title="PacketsLength non zero Mean", showfliers=False)

        tempDf = pd.DataFrame()
        for band in range(numBands):
            _sparcityFunc("u", "p", band)
            _sparcityFunc("d", "p", band)
        tempDf.plot.box(title="Sparcity", showfliers=False)
