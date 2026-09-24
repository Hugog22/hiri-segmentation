# HI&RI Datasets

This directory contains instructions for downloading the required datasets for the HI&RI project.

## Note on MSD Task03
DO NOT download the MSD Task03 dataset, as it is a duplicate of the LiTS dataset for the liver and tumor structures.

## 1. TotalSegmentator
TotalSegmentator provides full-body segmentation. We use it for liver and kidneys.
- **Link**: [Zenodo](https://zenodo.org/record/6802614)
- **Download command**:
```bash
wget https://zenodo.org/record/6802614/files/Totalsegmentator_dataset.zip
unzip Totalsegmentator_dataset.zip -d TotalSegmentator
```

## 2. LiTS (Liver Tumor Segmentation)
- **Link**: [CodaLab](https://competitions.codalab.org/competitions/17094)
- **Instructions**: Register on CodaLab and download the training data. Place the files in the `LiTS` directory.

## 3. KiTS23 (Kidney and Kidney Tumor Segmentation)
- **Link**: [GitHub](https://github.com/neheller/kits23)
- **Instructions**:
```bash
git clone https://github.com/neheller/kits23.git
cd kits23
python -m kits23.download_data --dest ../KiTS23
```

## 4. AMOS22 (Multi-Organ Segmentation)
- **Link**: [Zenodo](https://zenodo.org/record/7155725)
- **Instructions**: Download the AMOS22 dataset from Zenodo and extract it to the `AMOS22` directory.

## 5. BTCV (Beyond the Cranial Vault)
- **Link**: [Synapse](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
- **Instructions**: Register on Synapse.org, sign the Data Use Agreement (DUA), and download the dataset into the `BTCV` directory.

## Expected Directory Structure
```
data/
├── raw/
│   ├── TotalSegmentator/
│   ├── LiTS/
│   ├── KiTS23/
│   ├── AMOS22/
│   └── BTCV/
├── processed/
```

## License Summary Table
| Dataset | License | Terms |
| --- | --- | --- |
| TotalSegmentator | CC BY 4.0 | Free to share and adapt, must give appropriate credit. |
| LiTS | CC BY-NC-ND 4.0 | Non-commercial use, no derivatives. |
| KiTS23 | CC BY-NC-SA 4.0 | Non-commercial, share alike. |
| AMOS22 | CC BY 4.0 | Free to share and adapt, must give appropriate credit. |
| BTCV | Custom DUA | Requires Synapse registration and DUA signing. |
