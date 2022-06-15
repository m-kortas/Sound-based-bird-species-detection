# Sound-based bird recognition Web Application

## About the project

The project, created in 2020 by a group of women from the local Polish chapter of Women in Machine Learning & Data
Science (WiMLDS), was designed to be a training project & collaboration on a real-life problem which machine learning
can help to solve.

The web application has been developed by Magdalena Kortas (backend - Deep Learning, Python and Flask) and Aleksandra
Zachariasz (frontend - Flask, CSS, JS, HTML5 and graphic design).

Are you curious about how the AI solution has been built? Check
our **[Medium](https://towardsdatascience.com/sound-based-bird-classification-965d0ecacb2b)** article.

## Data download

Analysis and modelling of Polish birds songs. Recordings were downloaded from the

- **[xeno-canto.org](https://www.xeno-canto.org/)** which is a website dedicated to sharing bird sounds from all over
  the world (480k, September 2019).

Data can be dowloaded
using [this jupyter notebook file](https://github.com/wimlds-trojmiasto/birds/blob/master/notebooks/AM_downloadData.ipynb)
.

Bird have high interspecies variance - same bird species singing in different countries might sound completely
different. We searched for 33 classes of birds, but we used it _only_ if there was _over 50 recording per class_, we
used it for our training.
In total, we provide experiments with two ways:

#### 1) for 19 classes of birds - recorded in Poland, Germany, Slovakia, Czech and Lithuania.

- 1 . Found 418 files for Parusmajor ( 1 )
- 2 . Found 59 files for Passerdomesticus ( 3 )
- 3 . Found 107 files for Luscinialuscinia ( 4 )
- 4 . Found 111 files for Phoenicurusphoenicurus ( 7 )
- 5 . Found 446 files for Erithacusrubecula ( 8 )
- 6 . Found 80 files for Phoenicurusochruros ( 10 )
- 7 . Found 134 files for Sittaeuropaea ( 16 )
- 8 . Found 105 files for Alaudaarvensis ( 17 )
- 9 . Found 216 files for Phylloscopustrochilus ( 19 )
- 10 . Found 564 files for Turdusphilomelos ( 21 )
- 11 . Found 314 files for Phylloscopuscollybita ( 22 )
- 12 . Found 365 files for Fringillacoelebs ( 23 )
- 13 . Found 65 files for Sturnusvulgaris ( 24 )
- 14 . Found 329 files for Emberizacitrinella ( 25 )
- 15 . Found 58 files for Columbapalumbus ( 26 )
- 16 . Found 204 files for Troglodytestroglodytes ( 27 )
- 17 . Found 53 files for Cardueliscarduelis ( 30 )
- 18 . Found 97 files for Chlorischloris ( 31 )
- 19 . Found 667 files for Turdusmerula ( 33 )

#### 2) for 27 classes - recorded worldwide

## Data preprocessing

The data should be prepared. Each song is cut into 5 second recordings and preprocessed into melspectrograms. The
purpose is to normalize dataset to have same size along the whole dataset in one run, and to denoise recordings.
Morover, the data is filtered with a high-pass filter. Data can be preprocessed
using [this jupyter notebook file](https://github.com/wimlds-trojmiasto/birds/blob/master/notebooks/AM_prepareData.ipynb)
.

## Dataset split

[This file](https://github.com/wimlds-trojmiasto/birds/blob/master/notebooks/AM_splitDataset.ipynb) divides our dataset
into train, validation and test set in ratio 8:1:1. We can't use preprogrammed functions to do that, because we have
divided each of our files into other smallers (i.e. one sound to six images). Putting images made out of same mp3 file
might lead to the data leakage and make our results not trustworthy and biased.

## Training

We approached the problem of song classification with Convolutional Neural Networks. We have tested it with:

- Xception
- MobileNets
- EfficientNets
- Handcrafted CNN's
- Other
