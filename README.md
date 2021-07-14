# Data Scientist Nanodegree

It is mid 2021, and I am attending the Data Scientist Nanodegree in Machine Learning at [Udacity](udacity.com). This repository is for my work and for the 3 projects.

Here is the [original repository](https://github.com/udacity/intro-to-ml-tensorflow) from Udacity.

# Supervised Learning

Project: **Finding Donors for CharityML**

See result in the `finding_donors` folder. 

The work is done in the [`finding_donors.ipynb`](finding_donors/finding_donors.ipynb) notebook, and the finished result is found in the [`report.html`](finding_donors/report.html) file.

# Deep learning with Tensorflow

Project: **Image classification**

See result in the `image_classifier` folder.

Part 1 is in the [`Project_Image_Classifier_Project.ipynb`](image_classifier/Project_Image_Classifier_Project.ipynb) notebook, and the finished result is found in the [`report.html`](image_classifier/report.html) file.

Part 2 is done in the `predict.py` file. Try it like this:

```
cd image_classifier
python3 predict.py --category_names label_map.json --top_k 8 test_images/wild_pansy.jpg image_classifier.h5
```