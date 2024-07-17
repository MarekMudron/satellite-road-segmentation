# Road segmentation - POVa 2023 project

### Team members

* Marek Mudroň
* Tomáš Dubský
* Filip Osvald

### How to use
* Install requirements.txt by `pip install -r requirements.txt`
* Download [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset/) from Kaggle (requires login).
* Extract the dataset to data directory.
* Run `scrape_ma.sh` to download [Massachusetts Roads Dataset](https://www.cs.toronto.edu/~vmnih/data/) and merge it with the above dataset.

Directory should have following directory structure

```
├── data
│   ├── train
│   ├── train_mask_temp
│   ├── train_sat_temp
├── datasets
│   ├── combined_dataset.py
│   ├── deep_globe_dataset.py
│   ├── ma_dataset.py
├── models
│   └── runs
│       └── train
├── runs
│    └── train
│       ├── resnet18-3-both07:15:27.685865
│       ├── resnet18-5-both21:59:31.240246
│       ├── resnet18-5-both-pretrained23:04:41.642611
│       ├── resnet18-5-deepglobe-augmented-pretrained01:04:23.606854
│       └── resnet18-5-deepglobe-pretrained00:11:30.002388
├── doc.pdf
├── README.md
├── requirements.txt
├── resize_images.py
├── resnet18-5-both-pretrained
│   └── ...
├── run.py
├── runs
│   └── train
├── scrape_ma.sh
├── test.py
├── test-results.txt
├── test.sh
├── train.py
├── train.sh
└── utils
    └── early_stopper.py
```

# Training
To train desired model, use script `train.py` and specify desired arguments.

Our models were trained using `train.sh`.
You can find more info about it in `doc.pdf`.


To train model with 5 pretrained encoder layers on combined dataset use following command

`python3 train.py --train-data both  --encoder-depth 5 --pretrained-encoder`

# Testing
To train desired model, use script `test.py` and specify desired arguments.
Our models were trained using `test.sh`.

To test the model, provide path to the directory with weights and dataset, that you want your data to be tested on (both/deepglobe/ma)

`python3 test.py -m 'models/runs/train/resnet18-3-both07:15:27.685865'  --data both`


# Models
You can download models that we trained [here](https://drive.google.com/file/d/1-uPkFCuB3LwP5CKtALaixfGUT1pM6kYU/view?usp=sharing).