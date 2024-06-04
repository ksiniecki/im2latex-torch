


# Im2Latex

Deep CNN Encoder + LSTM Decoder with Attention for Image to Latex, the pytorch implemention of the model architecture used by the [Seq2Seq for LaTeX generation](https://guillaumegenthial.github.io/image-to-latex.html)


## Experimental results on the IM2LATEX-100K  test dataset

| BLUE-4 | Edit Distance | Exact Match |
| ------ | ------------- | ----------- |
| 40.80  | 44.23         | 0.27        |



## Getting Started



**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Download the dataset for training:**
```bash
cd data
wget https://im2markup.yuntiandeng.com/data/im2latex_validate_filter.lst
wget https://im2markup.yuntiandeng.com/data/im2latex_train_filter.lst
wget https://im2markup.yuntiandeng.com/data/im2latex_test_filter.lst
wget https://im2markup.yuntiandeng.com/data/formula_images_processed.tar.gz
wget https://im2markup.yuntiandeng.com/data/im2latex_formulas.norm.lst
tar -zxvf formula_images_processed.tar.gz
```

**Preprocess:**

```bash
python preprocess.py
```

**Build vocab**
```bash
python build_vocab.py
```

**Train:**
     python train.py --dropout=0.2 --add_position_features --epoches=25 --max_len=150
**Evaluate:**

```bash
python evaluate.py --split=test --model_path=ckpts/best_ckpt.pt --batch_size=32
```
**Predict:**
```bash
python .\predict.py --model_path ckpts/best_ckpt.pt --im_path data/formula_images_processed/1a00a76d4e.png
```
