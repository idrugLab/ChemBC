import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ------> hide info
import rdkit
import deepchem as dc
from rdkit import Chem
import numpy as np
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.utils.typing import RDKitMol
import joblib
import argparse
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile,VarianceThreshold
from rdkit.Chem.AtomPairs import Pairs
from tensorflow import keras
import pandas as pd
from sklearn.base import TransformerMixin

model_path = {
    'Bcap37':'./models/Bcap37.model',
    'BT-20':'./models/BT-20.model',
    'BT-474':'./models/BT-474.model',
    'BT-549':'./models/BT-549.model',
    'HS-578T':'./models/HS-578T.model',
    'MCF-7':'./models/MCF-7.model',
    'MDA-MB-231':'./models/MDA-MB-231.model',
    'MDA-MB-361':'./models/MDA-MB-361.model',
    'MDA-MB-435':'./models/MDA-MB-435.model',
    'MDA-MB-453':'./models/MDA-MB-453.model',
    'MDA-MB-468':'./models/MDA-MB-468.model',
    'SK-BR-3':'./models/SK-BR-3.model',
    'T-47D':'./models/T-47D.model',
    'HBL-100':'./models/HBL-100.model'
}



class model():

    def __init__(self,system,dataset_path):
        self.system = system
        self.dataset_path = dataset_path
        self.model_name = 'Morgan'
        # if self.system == 'BT-20' or self.system == 'HS-578T':
        #     self.model_name = 'rdkit'
        # elif self.system == 'BT-474':
        #     self.model_name = 'MACCS'
        # elif self.system == 'Bcap37':
        #     self.model_name = 'at'
        # else:
        #     self.model_name = 'Morgan'

    def load_dataset(self):# ----> load datasets
        # if self.model_name == 'MACCS':
        #     featurizer = dc.feat.MACCSKeysFingerprint()
        #     loader = dc.data.CSVLoader(tasks=[], feature_field="Smiles", featurizer=featurizer)  # smiles_field指smiles列的标签
        #     dataset_origin = loader.create_dataset(self.dataset_path, shard_size=8192)
        #     dataset = self.load(dataset_origin)
        #     return dataset
        # if self.model_name == 'rdkit':
        #     featurizer = dc.feat.RDKitDescriptors()
        #     loader = dc.data.CSVLoader(tasks=[], feature_field="Smiles", featurizer=featurizer)  # smiles_field指smiles列的标签
        #     dataset_origin = loader.create_dataset(self.dataset_path, shard_size=8192)
        #     dataset = self.feature_dataset(dataset_origin)
        #     transformer = dc.trans.MinMaxTransformer(transform_X=True, dataset=dataset)
        #     dataset = transformer.transform(dataset)
        #     return dataset
        # if self.model_name == 'at':
        #     featurizers = AtomPairFeaturizer()
        #     loader = dc.data.CSVLoader(tasks=[],feature_field="Smiles",featurizer=featurizers)
        #     dataset_origin = loader.create_dataset(self.dataset_path, shard_size=8192)
        #     dataset = self.load(dataset_origin)
        #     return dataset
        # if self.model_name == 'Morgan':
        featurizer = dc.feat.CircularFingerprint(size=1024)
        loader = dc.data.CSVLoader(tasks=[], feature_field="Smiles", featurizer=featurizer)  # smiles_field指smiles列的标签
        dataset_origin = loader.create_dataset(self.dataset_path, shard_size=8192)
        dataset = self.load(dataset_origin)
        return dataset
        
            



    def load_model(self):# ----> load model by joblib
        mp = model_path[self.system]
        #if self.system == 'SK-BR-3':
        #   reloaded = keras.models.load_model(mp)
        #    model = dc.models.MultitaskClassifier(1, 1024, layer_sizes=[1024, 1024], weight_decay_penalty=0.001,model_dir="s")
        #    model.model = reloaded
        #    return model
            
        #else:
        model = joblib.load(mp)
        return model

    def run(self):# ---- >predict scores
        model = self.load_model()
        datasets = self.load_dataset()
        print('模型已加载完毕，细胞系为：%s,文件名为：%s,输出文件名为：%s_result.csv'%(self.system,self.dataset_path,self.system))
        print('开始计算……')
        y_pred = model.predict(datasets)

        y_pre = y_pred[:, 1]
        data = pd.read_csv(self.dataset_path)
        data['Score'] = y_pre
        d = pd.DataFrame(data,index=None)
        d = d.sort_values(by="Score",ascending=False)
        d.to_csv('%s_result.csv'%self.system,index=None)
        print('计算完毕！')

    def load(self,dataset):
        x_load, y_load = [],[]
        for x, _, _, id in dataset.itersamples():
            x_load.append(x)
        dataset_new = dc.data.NumpyDataset(X=x_load)
        return dataset_new
    
    def feature_dataset(self,dataset):
        imp = SimpleImputer(strategy = "constant",fill_value=0)
        #imp =SimpleImputer(missing_values=np.nan, strategy='mean')
        
        sel = VarianceThreshold(threshold=(.7 * (1 - .7)))
        selector = SelectPercentile(percentile=30)
        x_load, y_load = [],[]
        import random
        for x, y, _, id in dataset.itersamples():
            x[x == np.inf] = np.nan
            x_load.append(x)
            y_load.append(int(random.randint(0,1)))
        x_load_1 = imp.fit_transform(x_load)
        x_load_2 = sel.fit_transform(x_load_1)
        x_load_3 = selector.fit_transform(x_load_2,y_load)
        y_load = np.array(y_load)[:, np.newaxis]
        dataset_new = dc.data.NumpyDataset(X=x_load_3)
        return dataset_new

class AtomPairFeaturizer(MolecularFeaturizer):
    def _featurize(self, mol: RDKitMol) -> np.ndarray:
        fp = list(Pairs.GetHashedAtomPairFingerprint(mol))
        fp = np.asarray(fp, dtype=float)
        return fp

def main(system,files):# ----> main process
    run = model(system,files)
    run.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str,default=None,
                        help='csv files with a coloum of smiles format')
    parser.add_argument('--system', type=str,default=None,
                        help='provide system of cells : Bcap37、BT-20、BT-474、BT-549、HS-578T、MCF-7、MDA-MB-231、MDA-MB-361、MDA-MB-435、MDA-MB-453、MDA-MB-468、SK-BR-3、T-47D、HBL-100')
    # parser.add_argument('--model', type=str,default=None,
    #                     help='provide models : RF_AtomPairs_H、RF_rdkit_H、RF_MACCS_H、RF_Morgan、RF_rdkit、DNN_Morgan')
    parser.add_argument('--all_systems', type=bool,default=False,
                        help='Choose all systems：defaul false')
    args = parser.parse_args()
    # try:
    #     main(args.files,args.system)
        
    # except:
    #     print('没有选择文件或者文件格式有错误。')
    all = args.all_systems
    if all:
        systems = ['Bcap37','BT-20','BT-474','BT-549','HS-578T','MCF-7','MDA-MB-231','MDA-MB-361','MDA-MB-435','MDA-MB-453','MDA-MB-468','SK-BR-3','T-47D','HBL-100']
        for s in systems:
            main(s,args.files)
            print()
    else:
        main(args.system,args.files)
    