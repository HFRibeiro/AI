#### Steps to create a custom object detection model from tensorflow legacy
##### Special thanks to Harrison from pythonprogramming.net, this files are just a merge from all his tutorial, check out in https://pythonprogramming.net
<br>

1. git clone https://github.com/SirRibeiro/AI.git
2. You will end up with a structure like this:
  ```bash
  .
  ├── data
  ├── generate_tfrecord.py
  ├── images
  │   ├── test (your testing images go here 20 min)
  │   └── train (your training images go here 150 min)
  ├── lbIMG
  ├── README.md
  ├── training
  └── xml_to_csv.py

  ```
3. Install some dependecies and open labelImg (Thanks tzutalin for this: source- https://github.com/tzutalin/labelImg):
 ```
sudo apt-get install pyqt5-dev-tools
sudo pip3 install lxml
make qt5py3
cd lbIMG
python3 labelImg.py
 ```
4. Once lableImg is open use 'open Dir' button and chose your directory, then press 'create RectBox' or 'w' and creat a rectangle on the part of the image that is of interest, than chose your lable and finaly 'save' (an XML file will be created) and 'next image', repeate this until you have your database set.

5. Converting xml to csv, install some more dependecies, and run xml_to_csv.py(Thanks to datitran for this: source- https://github.com/datitran/raccoon_dataset):
```
cd ..
pip install --user panda
python3 xml_to_csv.py
```

6. Now, grab generate_tfrecord.py. The only modification that you will need to make here is in the class_text_to_int function. You need to change this to your specific class.  If you had many classes, then you would need to keep building out this if statement.

  #### Replace this with label map
  ```
  def class_text_to_int(row_label):
      if row_label == 'lable1':
          return 1
      if row_label == 'lable2':
          return 1
      ...
      else:
          None
  ```

7. Next, in order to use this, we need to either be running from within the models directory of the cloned models github, or we can more formally install the object detection API.

  ```
  cd ~
  git clone https://github.com/tensorflow/models.git
  sudo apt-get install protobuf-compiler python-pil python-lxml
  sudo pip install jupyter
  sudo pip install matplotlib
  ```

8. From the tensorflow clone models/research/
  ```
  cd ~/models/research
  protoc object_detection/protos/*.proto --python_out=.
  export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/research/slim
  sudo python3 setup.py install
  ```
9. Finaly run:

  ```
  python3 generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record
  python3 generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record
  ```
  Now, in your data directory, you should have train.record and test.record.

10. Download pre-trained models and configuration files from tensorflow, and put then in /models/research/object_detection/legacy folder like this:
   ```
   cp -R ~/AI/training  ~/models/research/object_detection/legacy/training
   wget https://raw.githubusercontent.com/SirRibeiro/AI/master/ssd_mobilenet_v1_pets.config
   cd ..
   wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
   tar xvzf ssd_mobilenet_v1_coco_11_06_2017.tar.gz
   cp -R ~/AI/data  ~/models/research/object_detection/legacy/data
   cp -R ~/AI/images  ~/models/research/object_detection/legacy/images
   ```
#### Note in the 'ssd_mobilenet_v1_pets.config' wich is inside ~/models/research/object_detection/legacy/training have some importante things, the:
  ```
  num_classes: 1
  ```
#### Wich defines the number of classes that you selected and the:
  ```
  batch_size: 12
  ```
#### Wich is the batch size, if you get memory errors change this to a lower number, also keep in mind that the default config from tensorflow does not have our paths like:
  ```
  fine_tune_checkpoint: "ssd_mobilenet_v1_coco_11_06_2017/model.ckpt"
  input_path: "data/train.record"
  label_map_path: "data/object-detection.pbtxt"
  input_path: "data/test.record"
  label_map_path: "data/object-detection.pbtxt"
  ```
#### We have also copy training, data and images to tensorflow models, inside data we have the object-detection.pbtxt, which contains:
  ```
  item {
      id: 1
      name: 'lable1'
  }
  item {
      id: 2
      name: 'lable2'
  }
  ```
11. Starting the actual training
```
cd ~/models/research/object_detection/legacy/
python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config
```
#### You should see something like:
  ```
  INFO:tensorflow:global step 11788: loss = 0.6717 (0.398 sec/step)
  INFO:tensorflow:global step 11789: loss = 0.5310 (0.436 sec/step)
  INFO:tensorflow:global step 11790: loss = 0.6614 (0.405 sec/step)
  INFO:tensorflow:global step 11791: loss = 0.7758 (0.460 sec/step)
  INFO:tensorflow:global step 11792: loss = 0.7164 (0.378 sec/step)
  INFO:tensorflow:global step 11793: loss = 0.8096 (0.393 sec/step)
  ```
#### To check progress use:
  ```
  cd ~/models/research/object_detection/legacy/
  tensorboard --logdir='training'
  ```
#### Wich runs on http://127.0.0.1:6006 (visit in your browser)

12. Next we need to export the inference graph, we will use the export_inference_graph.py from models/research/object_detection/ but to keep everything together we will copy it to legacy, note that the XYYYYX, is the number of steps that your training as done, normaly around 12000 are enough

  ```
  cp ~/models/research/object_detection/export_inference_graph.py ~/models/research/object_detection/legacy/export_inference_graph.py

  python3 export_inference_graph.py \
      --input_type image_tensor \
      --pipeline_config_path training/ssd_mobilenet_v1_pets.config \
      --trained_checkpoint_prefix training/model.ckpt-XYYYYX \
      --output_directory mac_n_cheese_inference_graph
  ```
13. Final step, check if you can really detect something, for this I use a personal modified version of jupyter object_detection_tutorial.ipynb, wich is object_detection_tutorial.py to use this run:
  ```
  cd ~/models/research/object_detection/legacy/
  mkdir results
  python3 object_dection_tutorial.py
  ```
#### This will check your test_images folder inside the legacy folder from 1 to 57, like image1.jpg,image2.jpg...image57.jpg and will save the results to the results folder
