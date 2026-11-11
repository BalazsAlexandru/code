import hashlib
import json
import os
import random
import numpy as np
import cv2
import torch

from PIL import Image, ImageDraw
from Utils import BACKGROUND_RATIO, CACHE_PATH, PATCH_SIZE, SEED, STRIDE, DAMAGE_CLASSES, MAX_BUILDING_RATIO_FOR_BACKGROUND_PATCHES
from torch.utils.data import Dataset
from shapely import wkt

IMAGE_SIZE = 1024

class SegmentationDataset(Dataset):

    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.patchSize = PATCH_SIZE
        self.stride = STRIDE
        self.background_ratio = BACKGROUND_RATIO
        self.damage_classes = DAMAGE_CLASSES
        self.cachePath = CACHE_PATH
        self.maxBuildingRatioForBackgroundPatches = MAX_BUILDING_RATIO_FOR_BACKGROUND_PATCHES

        self.image_paths = sorted([
            os.path.join(self.images_dir, f)
            for f in os.listdir(self.images_dir)
            if f.endswith(".png") and "post_disaster" in f
        ])

        self.label_paths = sorted([
            os.path.join(self.labels_dir, f)
            for f in os.listdir(self.labels_dir)
            if f.endswith(".json") and "post_disaster" in f
        ])

        self.samples = []
        self._image_maps = {}
        self._mask_maps = {}

        # Validate that each image has a corresponding label and they are aligned correctly and there is an equal amount of images to labels
        assert len(self.image_paths) == len(self.label_paths), f"{len(self.image_paths)} images vs {len(self.label_paths)} labels"
        for images_path, labels_path in zip(self.image_paths, self.label_paths):
            img_stem = os.path.basename(images_path).replace(".png", "")
            lbl_stem = os.path.basename(labels_path).replace(".json", "")
            assert img_stem == lbl_stem, f"Mismatch: {img_stem} vs {lbl_stem}"

        buildingPatches = []
        backgroundPatches = []
        patchCenter = self.patchSize // 2
        maxAllowedBackgroundPixels = self.maxBuildingRatioForBackgroundPatches * PATCH_SIZE * PATCH_SIZE # Number of maximum allowed pixels for background patches

        for images_path, labels_path in zip(self.image_paths, self.label_paths):

            with open(labels_path, "r") as f:
                data = json.load(f)

            for feature in data["features"]["xy"]:
                damage = feature["properties"].get("subtype")
                if damage not in self.damage_classes:
                    continue

                xmin, ymin, xmax, ymax = wkt.loads(feature["wkt"]).bounds
                cx = int(round((xmin + xmax) / 2))
                cy = int(round((ymin + ymax) / 2))
                
                # Skip buildings too close to the border to form a full patch - ??? STILL CONSIDERING ??? #TODO
                if cx - patchCenter < 0 or cy - patchCenter < 0 or cx + patchCenter > IMAGE_SIZE or cy + patchCenter > IMAGE_SIZE:
                    continue
                
                buildingPatches.append((images_path, labels_path, cx, cy))

            #Create the background patches
            buildingPresenceMask = self.building_footprint_mask(data) # Generates a mask with the buildings, in order to extract background patches.

            # Integral image with a zero row/column so window sums need no edge cases
            integral = np.zeros((IMAGE_SIZE + 1, IMAGE_SIZE + 1), dtype=np.int64)
            integral[1:, 1:] = buildingPresenceMask.astype(np.int64).cumsum(axis=0).cumsum(axis=1)


            for y in range(patchCenter, IMAGE_SIZE - patchCenter + 1, self.stride):
                for x in range(patchCenter, IMAGE_SIZE - patchCenter + 1, self.stride):
                    x0, y0 = x - patchCenter, y - patchCenter
                    x1, y1 = x0 + PATCH_SIZE, y0 + PATCH_SIZE

                    building_pixels = (integral[y1, x1] - integral[y0, x1]
                                    - integral[y1, x0] + integral[y0, x0])

                    if building_pixels <= maxAllowedBackgroundPixels:
                        backgroundPatches.append((images_path, labels_path, x, y))


        backgroundPatchesNumber = int(len(buildingPatches) * (self.background_ratio / (1 - self.background_ratio)))
        backgroundPatchesNumber = min(backgroundPatchesNumber, len(backgroundPatches))
        sampledBackground = random.sample(backgroundPatches, backgroundPatchesNumber)
        print(f"Total background number: {backgroundPatchesNumber}")
        self.samples = buildingPatches + sampledBackground

        # --- Caching mechanism ---
        os.makedirs(self.cachePath, exist_ok=True)
        print(f"Cache directory:     {os.path.abspath(self.cachePath)}", flush=True)
        print(f"Cache files found:   {len(os.listdir(self.cachePath))}", flush=True)

        seen = set()
        for img_path, label_path, _, _ in self.samples:
            if img_path in seen:
                continue
            seen.add(img_path)

            imageCachePath = self._get_cache_path(img_path, True)
            if not os.path.exists(imageCachePath):
                with Image.open(img_path) as image:
                    np.save(imageCachePath, np.array(image.convert("RGB")))

            maskCachePath = self._get_cache_path(label_path, False)
            if not os.path.exists(maskCachePath):
                np.save(maskCachePath, self._build_mask(label_path))

        print(f"Pre-caching complete. {len(seen)} images processed.", flush=True)
        print(f"Cache files after:   {len(os.listdir(self.cachePath))}", flush=True)


    
    def __len__(self):
        return len(self.samples)



    def load_patch(self, idx):
        img_path, label_path, cx, cy = self.samples[idx]
        half = self.patch_size // 2
        x0, y0 = cx - half, cy - half
        x1, y1 = x0 + self.patch_size, y0 + self.patch_size

        image_map, mask_map = self._get_maps(img_path, label_path)
        image = np.array(image_map[y0:y1, x0:x1])    # copies just the patch out of the memmap
        mask = np.array(mask_map[y0:y1, x0:x1])
        return image, mask



    def __getitem__(self, idx):
        image, mask = self.load_patch(idx)
        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
        mask = torch.from_numpy(mask).long()
        return image, mask



    def _get_maps(self, img_path, label_path):
        if img_path not in self._image_maps:
            self._image_maps[img_path] = np.load(self._get_cache_path(img_path, True), mmap_mode="r")
        if label_path not in self._mask_maps:
            self._mask_maps[label_path] = np.load(self._get_cache_path(label_path, False), mmap_mode="r")
        return self._image_maps[img_path], self._mask_maps[label_path]



    def _get_cache_path(self, label_path, isImagePath):
        key = hashlib.md5(f"{label_path}_{IMAGE_SIZE}_{IMAGE_SIZE}".encode()).hexdigest()
        return os.path.join(self.cachePath, f"img_{key}.npy" if isImagePath == True else f"{key}.npy")



    @staticmethod
    def building_footprint_mask(data):
        #Rasterize the building into a binary mask
        mask = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
        draw = ImageDraw.Draw(mask) # Creates a 1024x1024 greyscale image filled with 0, plus a "pen" (draw) that paints onto it. "L" mode is 8 bit so each pixel can hold 0-255
        for feature in data["features"]["xy"]:
            geom = wkt.loads(feature["wkt"]) # For each building in the picture, convert the building polygon to a shapely geometry object
            for poly in SegmentationDataset._polygons(geom):
                draw.polygon(list(poly.exterior.coords), fill=1) # poly.exterior.coords gives the outer boundary as (x, y) points, which are already pixel coordinates (x = column, y = row), exactly what PIL expects. draw.polygon fills the inside with the value 1. Interior holes are ignored, which is fine for building footprints.
        return np.array(mask, dtype=np.uint8) # Converts the PIL image to a NumPy array of shape (1024, 1024) where 1 means “inside some building” and 0 means “not a building”. The mask includes all subtypes (also un-classified), because anything that is a building must not count as background.



    @staticmethod
    def _polygons(geom):
        return list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]



    def _build_mask(self, label_path):
        """Class-id mask (H, W) uint8 built from the 'xy' pixel polygons."""
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

        with open(label_path, "r") as f:
            data = json.load(f)

        buildings = [b for b in data["features"]["xy"]
                     if b["properties"].get("subtype") in self.damage_classes]
        # Draw lower class ids first so higher ids (e.g. destroyed) win where polygons overlap
        buildings.sort(key=lambda b: self.damage_classes[b["properties"]["subtype"]])

        # For each building, fill the mask with the respective classId values
        for building in buildings:
            classId = self.damage_classes[building["properties"]["subtype"]]
            geom = wkt.loads(building["wkt"])
            for poly in self._polygons(geom):
                coords = np.round(np.array(poly.exterior.coords, dtype=np.float64))
                coords = coords.astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [coords], int(classId))

        return mask



class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, mask = self.base.load_patch(self.indices[idx])   # uint8 numpy

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        if not isinstance(image, torch.Tensor):                  # transform without ToTensorV2
            image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.ascontiguousarray(mask))

        return image, mask.long()