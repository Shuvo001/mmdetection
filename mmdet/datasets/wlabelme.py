# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .wcustom import WCustomDataset
from itertools import count
from iotoolkit.labelme_toolkit import LabelMeData


@DATASETS.register_module()
class LabelmeDataset(WCustomDataset):

    def __init__(self,*args,**kwargs):
        self.img_suffix = kwargs.pop("img_suffix","jpg")
        self.classes = kwargs.get("classes")
        filter_empty_files = kwargs.pop("filter_empty_files",False)
        self.label_text2id = dict(zip(self.classes,count()))
        resample_parameters = kwargs.pop("resample_parameters",None)
        self.__dataset = LabelMeData(label_text2id=self.label_text2id,
                                     absolute_coord=True,
                                     filter_empty_files=filter_empty_files,
                                     resample_parameters=resample_parameters)
        super().__init__(*args,**kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        print(f"Load labelme annotations.")
        self.__dataset.read_data(ann_file,img_suffix=self.img_suffix)
        return self.__dataset