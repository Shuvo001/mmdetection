import numpy as np
from .builder import DATASETS
from .wcustom import WCustomDataset
from itertools import count
from iotoolkit.pascal_voc_toolkit import PascalVOCData


@DATASETS.register_module()
class WXMLDataset(WCustomDataset):

    def __init__(self,*args,**kwargs):
        self.img_suffix = kwargs.pop("img_suffix","jpg")
        self.classes = kwargs.get("classes")
        filter_empty_files = kwargs.pop("filter_empty_files",False)
        resample_parameters = kwargs.pop("resample_parameters",None)
        ignored_classes = kwargs.pop("ignored_classes",[])
        self.label_text2id = dict(zip(self.classes,count()))
        for c in ignored_classes:
            self.label_text2id[c] = None
        self.__dataset = PascalVOCData(label_text2id=self.label_text2id,
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
