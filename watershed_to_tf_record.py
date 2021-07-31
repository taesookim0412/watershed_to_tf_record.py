# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""
Convert your custom dataset to TFRecord for object_detection.

Base of this script is create_pet_tf_record.py provided by tensorflow repository on github
create_pet_tf_record.py could be found under tensorflow/models/research/object_detection/dataset_tools

Minimal Example usage:
  Python object_detection/dataset_tools/create_mask_rcnn_tf_record.py
    --data_dir_path=<path to directory containing dataset>
    --bboxes_provided=<True if you are providing bounding box annoations as xml file>
"""
import collections
import hashlib
import io
import logging
import os
import sys
from shutil import copyfile

import contextlib2
import numpy as np
import PIL.Image
import tensorflow.compat.v1 as tf
from lxml import etree

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

from cv2 import countNonZero
import collections
import json

def load_bbox(bbox_fp):
    '''bbox_fp: fp to bboxes json file
    returns: {label: [bboxes]}'''
    data = json.load(open(bbox_fp,))
    res = collections.defaultdict(list)
    for key, value in data.items():
        label = key[:key.find('[')]
        res[label] = value
    return res

def image_to_tf_data(img_path,
                     mask_path,
                     bbox_path,
                     label_map_dict,
                     filename):
    """Convert image and annotations to tf.tf_data proto.

    Note: if an image contains more than one object from same class
        then xmls files with bounding box annotation need to be provided

    Args:
      img_path: String specifying subdirectory within the dataset directory holding the actual image data.
      mask_path: String path to PNG encoded mask.
      xml_path: String path to XML file holding bounding box annotations
      label_map_dict: A map from string label names to integers ids.
      filename: Name of the image

    Returns:
      example: The converted tf.tf_data

    Raises:
      ValueError: if the image pointed to by filename is not a valid JPEG
    """
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    width = np.asarray(image).shape[1]
    height = np.asarray(image).shape[0]
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    with tf.gfile.GFile(mask_path, 'rb') as fid:
        encoded_mask_png = fid.read()
    encoded_png_io = io.BytesIO(encoded_mask_png)
    mask = PIL.Image.open(encoded_png_io)
    mask_np = np.asarray(mask.convert('L'))
    print("Uniques:")
    print(np.unique(mask_np))

    if mask.format != 'PNG':
        raise ValueError('Mask format not PNG')

    classes = []
    classes_text = []
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    encoded_mask_png_list = []
    areas = []
    is_crowds = []
    # if not os.path.exists(bbox_path):
    #     logging.warning('Could not find %s, ignoring example.', bbox_path)
    #     return
    if bbox_path is None:
        for key_name in label_map_dict.keys():
                pixel_val = int(label_map_dict[key_name][1])
                class_name = key_name + label_map_dict[key_name][1]
                nonbackground_indices_x = np.any(mask_np == pixel_val, axis=0)
                nonbackground_indices_y = np.any(mask_np == pixel_val, axis=1)
                nonzero_x_indices = np.where(nonbackground_indices_x)
                nonzero_y_indices = np.where(nonbackground_indices_y)

                if np.asarray(nonzero_x_indices).shape[1] > 0 and np.asarray(nonzero_y_indices).shape[1] > 0:
                    xmin = float(np.min(nonzero_x_indices))
                    xmax = float(np.max(nonzero_x_indices))
                    ymin = float(np.min(nonzero_y_indices))
                    ymax = float(np.max(nonzero_y_indices))
                    print(filename, 'bounding box for',
                          class_name,  xmin, xmax, ymin, ymax)

                    xmins.append(xmin / width)
                    ymins.append(ymin / height)
                    xmaxs.append(xmax / width)
                    ymaxs.append(ymax / height)

                    classes_text.append(class_name.encode('utf8'))
                    #idk which model uses 'classes'
                    #classes.append(label_map_dict[class_name][0])

                    mask_remapped = (mask_np == pixel_val).astype(np.uint8)
                    area = float(countNonZero(mask_remapped))
                    #print(f"area for {filename}: {area}")
                    areas.append(area)
                    is_crowds.append(0)
                    img = PIL.Image.fromarray(mask_remapped)
                    output = io.BytesIO()
                    img.save(output, format='PNG')
                    encoded_mask_png_list.append(output.getvalue())
    else:
        bbox_data = load_bbox(bbox_path)
        for class_name, bboxes in bbox_data.items():
            pixel_val = int(label_map_dict[class_name][1])
            #Our labelmap uses the class names with pix values at the end
            class_name = class_name + label_map_dict[class_name][1]
            for bbox in bboxes:
                ymin, xmin, ymax, xmax = bbox
                print(filename, 'bounding box for',
                      class_name,  xmin, xmax, ymin, ymax)
                xmins.append(xmin / width)
                ymins.append(ymin / height)
                xmaxs.append(xmax / width)
                ymaxs.append(ymax / height)
                classes_text.append(class_name.encode('utf8'))
                # classes.append(label_map_dict[class_name][0])

                mask_np_black = mask_np*0
                mask_np_black[int(ymin):int(ymax), int(xmin):int(
                    xmax)] = mask_np[int(ymin):int(ymax), int(xmin):int(xmax)]
                mask_remapped = (mask_np_black == pixel_val).astype(np.uint8)
                area = float(countNonZero(mask_remapped))
                areas.append(area)
                #mask_remapped = (mask_np == pixel_val).astype(np.uint8)

                if int(area) == 0:
                    print("Zero area: Bug found")
                    5/0
                is_crowds.append(0)
                img = PIL.Image.fromarray(mask_remapped)
                output = io.BytesIO()
                img.save(output, format='PNG')
                encoded_mask_png_list.append(output.getvalue())
    #(if no bboxes provided.. primitive])
    # else:
    #     for key in label_map_dict.keys():
    #         key_name = key
    #         pixel_val = int(label_map_dict[key_name][1])
    #         class_name = key_name + label_map_dict[key_name][1]
    #         nonbackground_indices_x = np.any(mask_np == pixel_val, axis=0)
    #         nonbackground_indices_y = np.any(mask_np == pixel_val, axis=1)
    #         nonzero_x_indices = np.where(nonbackground_indices_x)
    #         nonzero_y_indices = np.where(nonbackground_indices_y)
    #
    #         if np.asarray(nonzero_x_indices).shape[1] > 0 and np.asarray(nonzero_y_indices).shape[1] > 0:
    #             xmin = float(np.min(nonzero_x_indices))
    #             xmax = float(np.max(nonzero_x_indices))
    #             ymin = float(np.min(nonzero_y_indices))
    #             ymax = float(np.max(nonzero_y_indices))
    #             print(filename, 'bounding box for',
    #                   class_name,  xmin, xmax, ymin, ymax)
    #
    #             xmins.append(xmin / width)
    #             ymins.append(ymin / height)
    #             xmaxs.append(xmax / width)
    #             ymaxs.append(ymax / height)
    #
    #             classes_text.append(class_name.encode('utf8'))
    #             #idk which model uses 'classes'
    #             #classes.append(label_map_dict[class_name][0])
    #
    #             mask_remapped = (mask_np == pixel_val).astype(np.uint8)
    #             area = float(countNonZero(mask_remapped))
    #             #print(f"area for {filename}: {area}")
    #             areas.append(area)
    #             is_crowds.append(0)
    #             img = PIL.Image.fromarray(mask_remapped)
    #             output = io.BytesIO()
    #             img.save(output, format='PNG')
    #             encoded_mask_png_list.append(output.getvalue())

    feature_dict = {
        'image/height':             dataset_util.int64_feature(height),
        'image/width':              dataset_util.int64_feature(width),
        'image/filename':           dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id':          dataset_util.bytes_feature(filename.encode('utf8')),
        #?
        'image/key/sha256':         dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded':            dataset_util.bytes_feature(encoded_jpg),
        'image/format':             dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin':   dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax':   dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin':   dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax':   dataset_util.float_list_feature(ymaxs),
        #This text must be the same as the name in the labelmap!!!
        'image/object/class/text':  dataset_util.bytes_list_feature(classes_text),
        # 'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/mask':        dataset_util.bytes_list_feature(encoded_mask_png_list),
        'image/object/is_crowd':    dataset_util.int64_list_feature(is_crowds),
        'image/object/area':        dataset_util.float_list_feature(areas)}
    tf_data = tf.train.Example(
        features=tf.train.Features(feature=feature_dict))
    return tf_data


def create_tf_record(label_map_dict,
                     images_dir,
                     masks_dir,
                     bboxes_dir,
                     tfrecord_fullpath):
    """Creates a TFRecord file from data.

    Args:
      label_map_dict: The label map dictionary.
      images_dir: name of directory containing images
      masks_dir: name of directory containing masks corresponding to images
      bboxes_dir: name of directory containing bounding boxes annotation 
        correspoding to images
      tfrecord_fullpath: full path for tfrecord
    """
    with contextlib2.ExitStack() as tf_record_close_stack:
        # output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        #     tf_record_close_stack, tfrecord_path, FLAGS.num_shards)
        output_tfrecords = tf.python_io.TFRecordWriter(tfrecord_fullpath)

        image_names = [file for file in os.listdir(
            masks_dir) if file.endswith(".png")]
        for idx, filename in enumerate(image_names):
            base_name = os.path.splitext(filename)[0][:filename.find("_")]
            image_path = os.path.join(images_dir, base_name + ".jpg")
            mask_path = os.path.join(masks_dir, filename)
            bboxes_path = None
            if bboxes_dir is not None:
                bboxes_path = os.path.join(bboxes_dir, base_name + '_bbox.txt')
            try:
                tf_example = image_to_tf_data(image_path,
                                              mask_path,
                                              bboxes_path,
                                              label_map_dict,
                                              base_name)
                if tf_example:
                    #shard_idx = idx % FLAGS.num_shards
                    #output_tfrecords[shard_idx].write(
                    output_tfrecords.write(
                        tf_example.SerializeToString())
                    logging.info('done')
            except ValueError:
                logging.warning('Invalid example: %s, ignoring.', image_path)
    #copy the file from fullpath to datadir + tf filename
    #copyfile(tfrecord_path_datadir, tfrecord_fullpath)

def main(imgs_dir, masks_dir, bboxes_dir, label_map_fp, tfrecord_fp):
    '''args:
    imgs_dir: path to imgs,
    masks_dir: path to color_masks,
    bboxes_dir: path to bboxes,
    label_map_fp: full path to label map,
    tfrecord_fp: full path to tfrecord
    example labelmap located in label.pbtxt
    '''
    # read label mapping
    print('INFO: Reading label mapping')
    #{name+pix:id}
    label_map_dict = label_map_util.get_label_map_dict(
        label_map_fp)
    new_label_map_dict = collections.defaultdict(list)
    for key, value in label_map_dict.items():
        name, pix = key[:-3], key[-3:]
        new_label_map_dict[name] = [value, pix]
    #{name: [id, pix]}
    print("INFO: Label mapping: {a}".format(a=new_label_map_dict))
    # create tfrecord of data
    create_tf_record(new_label_map_dict,
                     imgs_dir,
                     masks_dir,
                     bboxes_dir,
                     tfrecord_fp)
