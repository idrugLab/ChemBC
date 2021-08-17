#Usage of ChemBC



##A. Configuration preparation:

​			1. rdkit

​			2. deepchem

​			3. sklearn

​			4. tensorflow



## B. Instructions:

​			1.--files xx.csv 
				File selection parameters, the file format refers to the sample format (D part).

​			2.--system xxx 
				Cell line selection parameters, select the cell line model provided by the application for prediction. (Bcap37, BT-20, BT-474, BT-549, HS-578T, MCF-7, MDA-MB-231, MDA-MB-361, MDA-MB-435, MDA-MB-453, MDA-MB-468, SK-BR-3, T-47D, and HBL-100)

​			3.--all_system True 
				All cell line parameters are selected. If this option is selected, there is no need to select the system parameter.

​			4. After the application is started, the corresponding scoring file will be automatically generated under the current path, in the format of csv.



## C. Examples:

​			 1. Select a single cell line:

```
	python ChemBC.py --files MCF-7.csv  --system HS-578T
```

​			 2. Select all cell lines:

```
	python ChemBC.py --files MCF-7.csv  --all True
```



## D. The format of the input file:

​			 1. The format of the input file should be csv.

​			 2. In the input file, the contents are as follows:

| Smiles                                                       |
| ------------------------------------------------------------ |
| O=C1CC[C@]2([C@@H](CC[C@@]3([C@@H]2CC[C@@H]2[C@@H]4[C@@](CC[C@]23C)(CC[C@H]4C(C)=C)C(=O)[O-])C)C1(C)C)C |
| O=C1CC[C@]2([C@@H](CC[C@@]3([C@@H]2CC[C@@H]2[C@@H]4[C@@](CC[C@]23C)(CC[C@H]4C(C)=C)C(OCCCC[NH+](CC)CC)=O)C)C1(C)C)C |
| O=C1CC[C@]2([C@@H](CC[C@@]3([C@@H]2CC[C@@H]2[C@@H]4[C@@](CC[C@]23C)(CC[C@H]4C(C)=C)C(OCCCC[NH+]2CCCC2)=O)C)C1(C)C)C |
| O=C1CC[C@]2([C@@H](CC[C@@]3([C@@H]2CC[C@@H]2[C@@H]4[C@@](CC[C@]23C)(CC[C@H]4C(C)=C)C(OCCCC[NH+]2CCCCC2)=O)C)C1(C)C)C |
| O1CC[NH+](CC1)CCCCOC(=O)[C@]12[C@@H]([C@H]3CC[C@H]4[C@](CC[C@@H]5[C@@]4(CCC(=O)C5(C)C)C)(C)[C@@]3(CC1)C)[C@@H](CC2)C(C)=C |
| O=C1CC[C@]2([C@@H](CC[C@@]3([C@@H]2CC[C@@H]2[C@@H]4[C@@](CC[C@]23C)(CC[C@H]4C(C)=C)C(OCCC[NH+](CC)CC)=O)C)C1(C)C)C |
| O=C1CC[C@]2([C@@H](CC[C@@]3([C@@H]2CC[C@@H]2[C@@H]4[C@@](CC[C@]23C)(CC[C@H]4C(C)=C)C(OCCC[NH+]2CCCC2)=O)C)C1(C)C)C |
| ……                                                           |

