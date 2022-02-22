from .base_dataset import BaseDataset


class ProbingDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        print("doing split ", split)
        assert split in ["train", "val", "test"]

        if split == "test":
            names = ["flickr_colors", "flickr_pos",  "flickr_size", "flickr_bshift", 
                    "flickr_colors_dec", "flickr_pos_dec",  "flickr_size_dec", "flickr_bshift_dec", 
                    "coco_altcap", "coco_altcap_dec", "flowers_flowers", 
                    "coco_objcount", "coco_objcount_dec", "flickr_tag", "flickr_tag_dec"]
        else:
            names=[]

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def __getitem__(self, index):
        return self.get_suite(index)
