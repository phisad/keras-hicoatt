# keras-hicoatt

An implementation of the hierarchical co-attention network for visual question answering from [Lu et. al (2016)](http://arxiv.org/abs/1606.00061) in Keras.

# Project Setup

### Clone and install the project

Clone the project to your machine

```
    git@github.com:phisad/thesis-starter.git
```

Install the project scripts by running 

```
    python setup.py install clean -a
```
 
If you want to install to a custom local directory, then create the site-packages directory and run the install script with a prefix. Afterwards you have to update the python path to make the custom directory available.

```
    mkdir -p $HOME/.local/lib/python3.5/site-packages
    python3 setup.py install --prefix=$HOME/.local
    export PYTHONPATH=$HOME/.local/lib/python3.5/site-packages:$PYTHONPATH
```

After installation you can run the following commands from the command line

```
    hicoatt-prepare-images
    hicoatt-prepare-questions
    hicoatt-session
```
   
The scripts will automatically look for a hicoatt-configuration.ini in the users home directory, but you can also specify a configuration file using the `-c` option.

### Prepare the configuration file

The commands require a configuration file to be setup. You can find a template file from within the egg file at

```
    $HOME/.local/lib/python3.5/site-packages/hicoatt/configuration/configuration.ini.template
```   
   
The recommendation is to copy this file to the user directory and rename it to 

```
    $HOME/hicoatt-configuration.ini
```
    
The configuration file describes attributes for the model, training and preparation.
    
# Image dataset preparation

### Setting up the dataset structure and configuration

Download the ZIP files (probably MSCOCO) and put them in a directory like

```
    ImageDatasetDirectoryPath = /data/mscoco
```
    
The project requires the following sub directory structure

```
     /data/mscoco
     +- train
     +- validate
     +- test
```

Extract the ZIP files to the according sub-directories. 

For MSCOCO the counts are

``` 
    NumberOfTrainingImages = 82783
    NumberOfValidationImages = 40504
    NumberOfTestImages = 81434
```

The image name infixes are 

```
    TrainingImageNameInfix = train2014
    ValidationImageNameInfix = val2014
    TestImageNameInfix = test2015
```

Put these information to the configuration file. The numbers are optional but useful to show preparation progress. 

The infixes are required to be specified. The image loade will look for files with the following naming pattern:

```
    COCO_<infix>_<imageid>
```
    
### Running the preparation script

The hicoatt paper uses the following configuration for the image preparation

```
    ImageInputShape = (448, 448, 3)
    ImageFeaturesSize = 196
```
 
Now you can run the preparation script. The script will resize the image to the configured image input shape, put them into a single tf records file and then compute the feature maps from them. The feature maps are directly put on hard disk as numpy files along with the source images.

```
    hicoatt-prepare-images all
```
 
# VQA dataset preparation

### Setting up the dataset structure and configuration

Download the ZIP files (probably VQA1.0) and put them in a directory like

    TextualDatasetDirectoryPath = /data/vqa1

The project requires the following sub directory structure

```
     /data/vqa1
     +- train
     +- validate
     +- test_dev
     +- test
```

Extract the ZIP files to the according sub-directories. Then rename the following to the following scheme:

```
    /data/vqa1
         +- train
         |   +- v1_OpenEnded_mscoco_questions.json
         |   +- v1_mscoco_annotations.json
         +- validate
         |   +- v1_OpenEnded_mscoco_questions.json
         |   +- v1_mscoco_annotations.json
         +- test_dev
         |   +- v1_OpenEnded_mscoco_questions.json
         +- test
             +- v1_OpenEnded_mscoco_questions.json
```
    
### Running the preparation script

The hicoatt paper uses the following configuration for the number of classes (answers)

```
    NumClasses = 1000
```
    
Now you can run the preparation script. The script will automatically prepare a train-validate split. For this purpose, the answers are fetched from the annotations file and the most common labels are determined. Then the training questions are filtered given the most common answers. Only the question that include one of the most common answers are kept. The remaining questions are fit to produce a training vocabulary. Given this vocabulary the maximal question length is determined. The produces files are put into the `/data/vqa1` diretory.

```
    hicoatt-prepare-questions all
```
    
For VQA 1.0 this will result in the following configuration properties for the train-validate split

```
    QuestionMaximalLength = 22
    VocabularySize = 12514
```

# Running the training session

When both image and question dataset are prepared, then the training session can be started.

The hicoatt paper uses the following configuration for training

```
    DropoutRate = 0.5
    ByPassImageFeaturesComputation = True
    ImageTopLayer = True
    Epochs = 256
    BatchSize = 300
```

You can also specify where to log the tensorboard events.

```
    TensorboardLoggingDirectory = /cache/tensorboard-logdir
```

Now you can run the training script.

```
    hicoatt-session training
```
    
The training will automatically prepare and start everything based on the configuration. 

You can also just check if everything is prepared with

```
    hicoatt-session training --dryrun
```
    
A checkpoint for further training can be specified with 

```
    hicoatt-session training -m <path-to-model> -i <epoch>
```

### A train+val session

The original paper did a training run on the training plus validation set. This can be easily prepared like the one above with the `-s trainval` option. This is also the option to start the training session.

# Running the evaluation session

The VQA 1.0 dataset requires a special result file to be uploaded. This result file is produced for the test-dev split with the following command

```
    hicoatt-session predict -m <path-to-model> -s "train test_dev"
```
    
The prediction takes the training split as a first argument in the split option to specify the vocabulary. Then as a second option the target split for prediction is specified. The result file is directly placed at the models path.

```
    <path-to-model>/vqa_OpenEnded_mscoco_test-dev2015_hicoatt_results.json
```
    
This file must be bundled to a `results.zip` and can then be uploaded to the VQA evaluation server.
    

 