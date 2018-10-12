# AI
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
