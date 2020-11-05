class Constants:
    SEED = 123
    DATA_DIR = '../data/' # Contains imgs/*.png, bboxes/*.pkl, and additional_features/*.pkl (optional) files
    SPLIT_DIR = 'splits'
    CLASS_NAMES = ['BG', 'Price', 'Title', 'Image']
    N_CLASSES = len(CLASS_NAMES)
    IMG_HEIGHT = 1280 # Image assumed to have same height and width
    OUTPUT_DIR = 'results_5-Fold_CV' # results dir is created here
