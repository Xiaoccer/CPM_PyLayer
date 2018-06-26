# CPM_PyLayer
This is the python implementation of the datalayer in [Convolutional Pose Machines release](https://github.com/shihenw/convolutional-pose-machines-release)

# Environments
* Python2.7
* Opencv2.4.13
* Numpy
* IPython

# Deitals
`data_transformer.py` is the implenentaiton like the [data_transformer.cpp](https://github.com/shihenw/caffe/blob/d154e896b48e8fb520cb4b47af8ba10bf9403382/src/caffe/data_transformer.cpp) in the Convolutional Pose Machines release code. But something is difference:
*   keep the method of data augmentation
*   reduce some params
*   remove the center map(if you want to detect mutli person, you can add it)

`cpm_data.py` shows how to use data_transformer.py in caffe.

`util.py` is the Visualization tools, produced by the [Convolutional Pose Machines release](https://github.com/shihenw/convolutional-pose-machines-release)

# How to use
* Replace the `data_dir` to your dataset
* Refer the data format in example folder to generate your dataset

# Cited Convolutional Pose Machine
    @inproceedings{wei2016cpm,
        author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
        booktitle = {CVPR},
        title = {Convolutional pose machines},
        year = {2016}
    }
