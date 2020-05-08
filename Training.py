from DataPreprocessing import DataProcessingClass

class ModelTraining():
    def __init__(self, dp: DataProcessingClass):
        self.data = dp.data

    def Model1(self):
        pass

    def Model2(self):
        pass




def main():
    df = pd.read_csv('data/train.tsv', sep='\t')
    dp = DataProcessingClass(df)
    mt = ModelTraining(dp.data)

if __name__ == '__main__':
    main()