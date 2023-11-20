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
                        index = root[root.index("\\")+1:]
                        toCsv.append([os.path.join(root, x), int(index)])
                        
                        
        toCsv.sort(key=lambda x: x[1])
        toCsv = [x for x in toCsv if x[1] <= self.i]

        with open(os.path.join(self.dir, self.csv), "w", newline='') as f:
            writer = csv.writer(f)
            for x in range(len(toCsv)):
                writer.writerow(toCsv[x])

    # def load(self):
    #     for root, files, name in os.walk(self.dir):
    #         if root != self.dir:
    #             index = root[root.index("\\")+1:]
    #             self.indexes.append(index)

    #     int_indexes = [int(i) for i in self.indexes]
    #     int_indexes.sort()
    #     self.indexes = [str(i) for i in int_indexes]

    #     for x in range(self.i):
    #         dir_path = os.path.join(self.dir, self.indexes[x])
    #         for root, files, name in os.walk(dir_path):
    #             for file in name:
    #                 im = Image.open(os.path.join(dir_path, file))
    #                 # img_copy = img.copy()
    #                 g = Grayscale(num_output_channels=1)
    #                 im = g(im)
    #                 img = np.asarray(im, dtype='float32')
    #                 self.dataset.append(img)
    #                 self.labels.append(x+1)
    
    # def getData(self):
    #     return self.dataset, torch.from_numpy(np.asarray(self.labels, dtype='float32'))