import os
import csv

class DatasetLoad():
    ## PARAMS
    # dir - root directory of image folders
    # i - maximum character index to include in csv
    # csvFile - filename of csv file
    def __init__(self, dir, i, csvFile):
        self.dir = dir
        self.i = i
        self.csv = csvFile
        
    def createCsv(self):
        toCsv = []
        for root, files, name in os.walk(self.dir):      
            if root != self.dir:
                for x in name:
                    if '.png' in x:      
                        # print(root)          
                        index = root[root.index("\\")+1:]
                        print(index)
                        toCsv.append([os.path.join(root, x), int(index)])
                        
                        
        toCsv.sort(key=lambda x: x[1])
        toCsv = [x for x in toCsv if x[1] <= self.i]
        # print(toCsv)

        with open(os.path.join(self.dir, self.csv), "w+", newline='') as f:
            writer = csv.writer(f)
            for x in range(len(toCsv)):
                writer.writerow(toCsv[x])
        
            f.close()

        return toCsv


ds = DatasetLoad('./source', 2, 'trainData.csv').createCsv()
print(ds)
