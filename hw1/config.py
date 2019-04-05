from yacs.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Define Config Node
# ---------------------------------------------------------------------------- #
_C = CN()

# ---------------------------------------------------------------------------- #
# Models
# ---------------------------------------------------------------------------- #
_C.MODELS = CN()
_C.MODELS.ARCH_DCIT = [{
    'behavioral_cloning': {
        'input_layer': {
            'units': 128,
            'activation': 'relu'
        },
        'hidden_layers': {
            'layer1': {
                'units': 128,
                'activation': 'relu'
            },
            'layer2': {
                'units': 64,
                'activation': 'relu'
            },
        },
        'output_layer': {
            'activation': 'linear'
        },
        'optimize': {
            'loss': 'mse',
            'optimizer': 'adam',
            'metrics': ['accuracy']
        },
        'train': {
            'batch_size': 64,
            'epochs': 40,
            'verbose': 1,
        },
    },
    'dagger': {
        'input_layer': {
            'units': 128,
            'activation': 'relu'
        },
        'hidden_layers': {
            'layer1': {
                'units': 128,
                'activation': 'relu'
            },
            'layer2': {
                'units': 64,
                'activation': 'relu'
            },
        },
        'output_layer': {
            'activation': 'linear'
        },
        'optimize': {
            'loss': 'mse',
            'optimizer': 'adam',
            'metrics': ['accuracy']
        },
        'train': {
            'batch_size': 64,
            'epochs': 40,
            'verbose': 1,
        },
    },
}]

_C.MODELS.MODELS_DIR = [{
    'behavioral_cloning': {
        'dir': './bc_models',
        'suffix': '_bc.h5',
    },
    'dagger': {
        'dir': './dagger_models',
        'suffix': '_dagger.h5'
    },
}]

_C.MODELS.DAGGER_LOOPS = 5

# ---------------------------------------------------------------------------- #
# DATA
# ---------------------------------------------------------------------------- #
_C.DATA = CN()
_C.DATA.TEST_TRAIN_RATIO = 0.1


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
