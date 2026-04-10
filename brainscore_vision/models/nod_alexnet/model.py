import torchvision.models as models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry

MODEL_ID = "alexnet_nod"

def get_model():
    pt_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    preprocessing = load_preprocess_images(image_size=224)
    wrapper = PytorchWrapper(
        model=pt_model,
        preprocessing=preprocessing,
        identifier=MODEL_ID,
        batch_size=16,
        device="cuda" if False else "cpu",  # force CPU here
    )
    return wrapper

def get_layers():
    # fixed list of candidate layers, no layer search at benchmark time
    return [
        "features.2",   # conv2
        "features.5",   # conv3
        "features.10",  # conv4
        "features.12",  # conv5
    ]

# register in model registry in a separate module
def register():
    model_registry[MODEL_ID] = lambda: ModelCommitment(
        identifier=MODEL_ID,
        activations_model=get_model(),
        layers=get_layers(),
    )