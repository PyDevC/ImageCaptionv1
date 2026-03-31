import os
import kagglehub

os.environ['KAGGLEHUB_CACHE'] = os.path.join(os.path.dirname(__file__), 'COCO')
path = kagglehub.dataset_download("riffaap/mscoco-dataset")
print(path)
