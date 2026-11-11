import random
import numpy as np
import torch
import albumentations
import tqdm

# --- Parameters ---
SEED = 67
IMAGES_TRAIN_PATH = r"C:\Users\40754\Desktop\Disertation\Dataset\TrainSmall\images"
LABELS_TRAIN_PATH = r"C:\Users\40754\Desktop\Disertation\Dataset\TrainSmall\labels"
CACHE_PATH = r"C:\Users\40754\Desktop\Disertation\Dataset\Cache"
PATCH_SIZE = 256
STRIDE = 190
BACKGROUND_RATIO = 0.2
CACHE_PATH = r"C:\Users\40754\Desktop\Disertation\Dataset\CacheXBD"
DAMAGE_CLASSES = {"no-damage": 1, "minor-damage": 2, "major-damage": 3, "destroyed": 4}
WEIGHTS = [0.2, 1.0, 2.0, 3.0, 4.0]
MODEL_NAME = "unet++" # Possible values: unet++, deeplabv3+, deeplabv3_resnet50, deeplabv3_resnet101
MAX_BUILDING_RATIO_FOR_BACKGROUND_PATCHES = 0.05
MEAN_NORMALIZATION_TRANSFORM = (0.485, 0.456, 0.406)
STD_NORMALIZATION_TRANSFORM  = (0.229, 0.224, 0.225)
BATCH_SIZE = 16
ADAM_LEARNING_RATE = 1.5e-4
EPOCHS = 100
RESUME_FROM = None
PATIENCE = 10

# --- Fixes the seed to all the random number generators that PyTorch uses, for reproducibility purposes ---
def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # ! Switch them to False and True if you want to use the GPU for training, but it will make the results non-deterministic
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# --- Transforms ---
segmentationTrainingAugmentation = albumentations.Compose([
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.RandomRotate90(p=0.5),
    albumentations.Normalize(mean=MEAN_NORMALIZATION_TRANSFORM, std=STD_NORMALIZATION_TRANSFORM, max_pixel_value=255.0),
    albumentations.ToTensorV2()
])

segmentationTrainingEvaluationAugmentation = albumentations.Compose([
    albumentations.Normalize(mean=MEAN_NORMALIZATION_TRANSFORM, std=STD_NORMALIZATION_TRANSFORM, max_pixel_value=255.0),
    albumentations.ToTensorV2()
])

# --- Validation function ---
def validateModel(model, validationLoader, device, epoch, criterion):
    model.eval()
    totalLoss = 0.0

    valuationLoss = 0.0
    with torch.no_grad():
        for images, labels in validationLoader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            totalLoss += loss.item()

    averageLoss = totalLoss / len(validationLoader)
    return averageLoss


def forward_pass(model, images):
    outputs = model(images)
    if isinstance(outputs, dict):          # torchvision models return {"out": ...}
        outputs = outputs["out"]
    return outputs
 
 
@torch.no_grad()
def validateModel(model, loader, device, criterion, num_classes=5):
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    totalLoss = 0.0
 
    for images, masks in tqdm(loader, desc="Validation", leave=False):

        images, masks = images.to(device), masks.to(device).long()
 
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = forward_pass(model, images)
        outputs = outputs.float()
 
        totalLoss += criterion(outputs, masks).item() * images.size(0)
        preds = outputs.argmax(dim=1)
 
        # count every (true class, predicted class) pair of the batch in one go
        pairs = (masks * num_classes + preds).flatten()
        confusion += torch.bincount(pairs, minlength=num_classes ** 2).view(num_classes, num_classes)
 
    evaluationLoss = totalLoss / len(loader.dataset)
 
    tp = confusion.diag().float()
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp
    iou = tp / (tp + fp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    mean_iou = iou[1:].nanmean().item()
 
    print(f"Val loss {evaluationLoss:.4f} | mean IoU (damage classes) {mean_iou:.4f}")
    for c in range(num_classes):
        print(f"  class {c}: IoU {iou[c]:.4f}  F1 {f1[c]:.4f}")
 
    return evaluationLoss, mean_iou