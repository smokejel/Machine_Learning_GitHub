from fastai import *
from fastai.vision import *

# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')

def obtain_data(dest):
    '''Single statement to download URL.PETS data to a path'''
    path = untar_data(URLs.PETS, None, dest)
    print(path)
    path_anno = (str(path) + "/annotations")
    path_img = (str(path) + "/images")

    fnames = get_image_files(path_img)

    np.random.seed(2)
    pattern = r'/([^/]+)_\d+.jpg$'

    data = ImageDataBunch.from_name_re(path_img, fnames, pattern, ds_tfms=get_transforms(), size=224, bs=64)
    data.normalize(imagenet_stats)
    return data

def train(data):
    learn = cnn_learner(data, models.resnet34, metrics=error_rate)
    learn.fit_one_cycle(4)
    learn.save('stage-1', return_path=True)
    return learn

def main_function():
    # path = 'C:/Users/mdswe/PycharmProjects/Machine_Learning/oxford-iiit-pet'
    # path_anno = 'C:/Users/mdswe/PycharmProjects/Machine_Learning/oxford-iiit-pet/annotations'
    path_img = 'C:/Users/mdswe/PycharmProjects/Machine_Learning_GitHub/FastAI'
    data = obtain_data(path_img)
    model = train(data)
    model.load('stage-1')
    interp = ClassificationInterpretation.from_learner(model)
    interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
    plt.show()

if __name__ == '__main__':
    main_function()
