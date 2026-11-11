import segmentation_models_pytorch
import torch
import torchvision


def getModel(modelName, numClasses):

    if modelName == "unet++":
        return segmentation_models_pytorch.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=numClasses
        )

    elif modelName == "deeplabv3+":
        return segmentation_models_pytorch.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=numClasses
        )

    elif modelName == "deeplabv3_resnet50":
        model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None)
        model.classifier[-1] = torch.nn.Conv2d(
            256,
            numClasses,
            kernel_size=1
        )
        return model

    elif modelName == "deeplabv3_resnet101":
        model = torchvision.models.segmentation.deeplabv3_resnet101(weights=None)
        model.classifier[-1] = torch.nn.Conv2d(
            256,
            numClasses,
            kernel_size=1
        )
        return model

    else:
        raise ValueError(
            f"Unknown model: {modelName}"
        )