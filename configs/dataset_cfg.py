
def get_img_size(dataset_name: str):

    if dataset_name == 'allweather':
        img_size = (480, 720)
    elif dataset_name == 'cityscapes':
        img_size = (1024, 2048)
    elif dataset_name in ['raindrop', 'snow100k', 'outdoor_rain']:
        img_size = (480, 720)
    elif dataset_name == 'synthetic_rain':
        img_size = (320, 480)
    elif dataset_name == 'ots':
        img_size = (640, 640)
    elif dataset_name == 'RVSD':
        img_size = (480,720)
    else:
        raise NotImplementedError

    return img_size

def get_crop_ratio(dataset_name: str):
    
    if dataset_name == 'allweather':
        crop_ratio = (1.0/2.0, 1.0/3.0)
    elif dataset_name == 'cityscapes':
        crop_ratio = (1.0/4.0, 1.0/8.0)
        # crop_ratio = (1.0/2.0, 1.0/4.0)
    elif dataset_name in ['raindrop', 'snow100k', 'synthetic_rain', 'outdoor_rain']:
        crop_ratio = (1.0/2.0, 1.0/3.0)
    elif dataset_name == 'ots':
        crop_ratio = (1.0, 1.0)
    elif dataset_name == 'RVSD':
        crop_ratio = (1.0/2.0, 1.0/3.0)
    else:
        raise NotImplementedError

    return crop_ratio


def get_dataset_root(dataset_name: str):
    
    if dataset_name == 'allweather':
        root = 'data/allweather'
    elif dataset_name == 'cityscapes':
        raise NotImplementedError
    elif dataset_name == 'raindrop':
        raise NotImplementedError
    elif dataset_name == 'snow100k':
        raise NotImplementedError
    elif dataset_name == 'synthetic_rain':
        raise NotImplementedError
    elif dataset_name == 'outdoor_rain':
        raise NotImplementedError
    elif dataset_name == 'ots':
        raise NotImplementedError
    elif dataset_name == 'RVSD':
        root = 'data/RVSD'
    else:
        raise NotImplementedError

    return root

def get_no_val_dataset(): 
    return ['allweather', 'raindrop', 'snow100k', 'synthetic_rain', 'outdoor_rain', 'ots', 'RVSD']