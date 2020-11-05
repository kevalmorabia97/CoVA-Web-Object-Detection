class Constants:
    SEED = 123 # in hope of achieving reproducible results
    DATA_DIR = '../data/' # contains imgs/*.png, bboxes/*.pkl, and additional_features/*.pkl (optional)
    SPLIT_DIR = 'splits' # contains Fold-%d dir containing {train|val|test}_{imgs|domains}.txt and webpage_info.csv (optional)
    CLASS_NAMES = ['BG', 'Price', 'Title', 'Image'] # Accuracies of class-0 (BG) are ignored
    N_CLASSES = len(CLASS_NAMES)
    IMG_HEIGHT = 1280 # image assumed to have same height and width
    OUTPUT_DIR = 'results_5-Fold_CV' # results dir is created here
