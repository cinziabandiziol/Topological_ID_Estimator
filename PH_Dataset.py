import numpy as np

class PH_Dataset:
    def __init__(self, kind, name, PH_dim, number_str):
        self.kind = kind
        self.name = name
        self.PH_dim = PH_dim
        self.number_str = number_str

    def CreateLoadDataset(self):

        if self.kind == 'Benchmark':

            filename = './PH/' + self.kind + '/' + self.name + self.number_str + '_PH' + str(self.PH_dim) + '.npy'

        elif self.kind == 'Other':

            # Reactgo_filtered, Reactgo, Fgdo, Contextdm1
            filename = './PH/' + self.kind + '/' + self.name + self.number_str + '_PH' + str(self.PH_dim) + '.npy'      

        elif self.kind == 'Neuro':

            # Reactgo_filtered, Reactgo, Fgdo, Contextdm1
            filename = './PH/' + self.kind + '/' + self.name + self.number_str + '_PH' + str(self.PH_dim) + '.npy'

        else:

            raise TypeError('Unknown dataset!!')

        PH = np.load(filename, allow_pickle = True)

        return PH
    
