import pytorch_lightning as pl
import json
from tqdm import tqdm
import pandas as pd

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, path='data', num_workers=0, pin_memory=False, train_trans=None, val_trans=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.path = path
        self.train_trans = train_trans
        self.val_trans = val_trans

    # read json file with annotations
    def load_annotations(self, mode='train'):
        instances_path = f'{self.path}/annotations/instances_{mode}2017.json'
        with open(instances_path) as json_file:
            return json.load(json_file)

    # load annotations for each image
    def process_annotations(self, instances):
        image_ids = [img['id'] for img in instances['images']]
        annotations = [[] for image_id in image_ids]
        for anns in tqdm(instances['annotations']):
            image_id = anns['image_id']
            category = anns['category_id']
            bbox = anns['bbox']
            area = anns['area']
            # we build a list of bbs and classes (and area)
            annotations[image_ids.index(image_id)].append(
                (bbox, category, area))
        # sort annotations by size and drop size
        sorted_anns = []
        for anns in annotations:
            sorted_anns.append([ans[:2] for ans in sorted(anns, key=lambda x: x[-1], reverse=True)])
        # remove images without labels
        final_ids, final_annotations = [], []
        for image_id, anns in zip(image_ids, sorted_anns):
            if len(anns) > 0:
                final_ids.append(image_id)
                final_annotations.append(anns)
        return final_ids, final_annotations

    # save processed annotations in json file to avoid processing every time
    def save_processed_annotations(self, mode, final_ids, instances, annotations, file_path):
        data = pd.DataFrame({
            'image_path': [f'{self.path}/{mode}2017/{image["file_name"]}' for image in instances['images'] if image['id'] in final_ids],
            'annotations': annotations
        })
        data.to_json(file_path)
        return data

    def setup(self, stage=None):
        # try to load annotations from processed files
        # else, make files
        self.data = {}
        for mode in ['train', 'val']:
            processed_data_file_name = f'{self.path}/{mode}2017_processed.json'
            try:
                self.data[mode] = pd.read_json(processed_data_file_name)
            except:
                annotations = self.load_annotations(mode)
                final_ids, processed_annotations = self.process_annotations(annotations)
                self.data[mode] = self.save_processed_annotations(
                    mode, final_ids, annotations, processed_annotations, processed_data_file_name)
        # try to load categories
        # else, make them
        categories_path = f'{self.path}/categories.json'
        try:
            with open(categories_path) as json_file:
                self.classes = json.load(json_file)
        except:
            self.classes = self.load_annotations()['categories']
            with open(categories_path, 'w') as outfile:
                json.dump(self.classes, outfile)
