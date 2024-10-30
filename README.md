# Deep Neural Network for Staging Adenocarcinoma via Multimodal Fusion of Lung CT Images, Coarse Annotation Bounding Boxes, and Electronic Health Records
![](fig/framework.png)

## Datasets
**The private LUNA-M dataset** collected from the Cancer Hospital & Shenzhen Hospital (Shenzhen, China). This dataset comprises 1,614 cases of lung adenocarcinoma from 1,430 anonymized patients, with each case including CT scans, clinical data, and bounding boxes identifying tumor locations. The cases are categorized into IA, MA, and AIS, with class proportions of 53.5\%, 24.1\%, and 22.4\%, respectively.

**The open-source LPCD dataset** derived from [Lung-PET-CT-Dx](https://doi.org/10.7937/TCIA.2020.NNC2-0461), contains the same modalities as LUNA-M. It includes 342 lung cancer cases, categorized into adenocarcinoma, small cell carcinoma, large cell carcinoma, and squamous cell carcinoma. Due to the limited squamous cell cases, the large and squamous cell categories are merged, resulting in a three-class dataset with distributions of 70.7\%, 17.3\%, and 12.0\%, respectively. 

## Start
train: `python train_clip.py --lr 0.001 --batch-size 24 --epochs 200 --phase train`

eval: `python train_clip.py --lr 0.001 --batch-size 24 --epochs 200 --MODEL-WEIGTH model.pt --phase val`
