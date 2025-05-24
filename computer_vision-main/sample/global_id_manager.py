# # global_id_manager.py
# import torch
# import torchvision.transforms as T
# import numpy as np
# from torchreid.utils import FeatureExtractor
# from scipy.spatial.distance import cosine

# class GlobalIDManager:
#     def __init__(self, threshold=0.5):
#         self.extractor = FeatureExtractor(
#             model_name='osnet_x1_0',
#             model_path='',  # Leave blank to auto-download
#             device='cuda' if torch.cuda.is_available() else 'cpu'
#         )
#         self.known_features = []
#         self.global_id_counter = 0
#         self.threshold = threshold

#         self.transform = T.Compose([
#             T.ToPILImage(),
#             T.Resize((256, 128)),
#             T.ToTensor(),
#             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])

#     def assign_global_id(self, crop):
#         if crop.shape[0] == 0 or crop.shape[1] == 0:
#             return -1

#         img = self.transform(crop).unsqueeze(0)
#         feature = self.extractor(img)[0]  # Extract feature vector

#         # Compare with known features
#         for idx, known_feat in enumerate(self.known_features):
#             if cosine(feature, known_feat) < self.threshold:
#                 return idx

#         # New identity
#         self.known_features.append(feature)
#         self.global_id_counter += 1
#         return self.global_id_counter - 1

# import torch
# import torchvision.transforms as T
# import numpy as np
# import os
# import cv2
# from torchreid.utils import FeatureExtractor
# from scipy.spatial.distance import cosine

# class GlobalIDManager:
#     def __init__(self, threshold=0.5):
#         self.extractor = FeatureExtractor(
#             model_name='osnet_x1_0',
#             model_path='',  # Leave blank to auto-download
#             device='cuda' if torch.cuda.is_available() else 'cpu'
#         )
#         self.known_features = []
#         self.global_id_counter = 0
#         self.threshold = threshold

#         self.transform = T.Compose([
#             T.ToPILImage(),
#             T.Resize((256, 128)),
#             T.ToTensor(),
#             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])

#         os.makedirs('reid_data', exist_ok=True)

#     def assign_global_id(self, crop):
#         if crop is None or crop.size == 0 or np.mean(crop) < 5:
#             print("[⚠️] Skipping blank or invalid crop.")
#             return -1

#         try:
#             img = self.transform(crop).unsqueeze(0)
#         except Exception as e:
#             print(f"[❌] Transform failed: {e}")
#             return -1

#         try:
#             feature = self.extractor(img)[0]
#         except Exception as e:
#             print(f"[❌] Feature extraction failed: {e}")
#             return -1

#         # Compare with known features
#         for idx, known_feat in enumerate(self.known_features):
#             dist = cosine(feature, known_feat)
#             if dist < self.threshold:
#                 print(f"[🔁] Matched with ID {idx} (cosine dist: {dist:.4f})")
#                 return idx

#         # New identity
#         print(f"[🆕] New ID {self.global_id_counter} assigned.")
#         self.known_features.append(feature)
#         cv2.imwrite(f'reid_data/crop_{self.global_id_counter}.jpg', crop)
#         self.global_id_counter += 1
#         return self.global_id_counter - 1

# global_id_manager.py
# import torch
# import torchvision.transforms as T
# import numpy as np
# import os
# import cv2
# from torchreid.utils import FeatureExtractor
# from scipy.spatial.distance import cosine

# class GlobalIDManager:
#     def __init__(self, threshold=0.3):
#         self.extractor = FeatureExtractor(
#             model_name='osnet_x1_0',
#             model_path='',  # Leave blank to auto-download
#             device='cuda' if torch.cuda.is_available() else 'cpu'
#         )
#         self.known_features = []
#         self.global_id_counter = 0
#         self.threshold = threshold
#         self.frame_id = 0  # to track frame count for filenames

#         os.makedirs("reid_data", exist_ok=True)

#         self.transform = T.Compose([
#             T.ToPILImage(),
#             T.Resize((256, 128)),
#             T.ToTensor(),
#             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])

#     def assign_global_id(self, crop):
#         if crop.shape[0] == 0 or crop.shape[1] == 0:
#             return -1

#         img = self.transform(crop).unsqueeze(0)
#         feature = self.extractor(img)[0]

#         for idx, known_feat in enumerate(self.known_features):
#             dist = cosine(feature, known_feat)
#             if dist < self.threshold:
#                 print(f"[🔁] Matched with ID {idx} (cosine dist: {dist:.4f})")
#                 self.save_crop(crop, idx)
#                 return idx

#         self.known_features.append(feature)
#         self.global_id_counter += 1
#         print(f"[🆕] New ID {self.global_id_counter - 1} assigned.")
#         self.save_crop(crop, self.global_id_counter - 1)
#         return self.global_id_counter - 1

#     def save_crop(self, crop, gid):
#         filename = f"reid_data/crop_gid{gid}_frame{self.frame_id}.png"
#         cv2.imwrite(filename, crop)
#         self.frame_id += 1

import torch
import torchvision.transforms as T
import numpy as np
import os
import cv2
from torchreid.utils import FeatureExtractor
from scipy.spatial.distance import cosine


class GlobalIDManager:
    def __init__(self, threshold=0.3):
        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='',  # Leave blank to auto-download
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.known_features = []
        self.global_id_counter = 0
        self.threshold = threshold
        self.frame_id = 0  # to track crop filenames

        os.makedirs("reid_data", exist_ok=True)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def assign_global_id(self, crop, camera_id="cam0"):
        """
        Assigns a global ID to the input crop image.

        Args:
            crop (np.ndarray): Cropped image of a detected person.
            camera_id (str): Identifier for the source camera or video.

        Returns:
            int: Global ID assigned to this person.
        """
        if crop is None or crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            print(f"[⚠️] Skipping blank or invalid crop from {camera_id}.")
            return -1

        try:
            img = self.transform(crop).unsqueeze(0)
        except Exception as e:
            print(f"[❌] Transform failed from {camera_id}: {e}")
            return -1

        try:
            feature = self.extractor(img)[0]
        except Exception as e:
            print(f"[❌] Feature extraction failed from {camera_id}: {e}")
            return -1

        for idx, known_feat in enumerate(self.known_features):
            dist = cosine(feature, known_feat)
            if dist < self.threshold:
                print(f"[🔁] Matched with Global ID {idx} from {camera_id} (cosine dist: {dist:.4f})")
                self.save_crop(crop, idx, camera_id)
                return idx

        self.known_features.append(feature)
        assigned_id = self.global_id_counter
        self.global_id_counter += 1
        print(f"[🆕] New Global ID {assigned_id} assigned from {camera_id}.")
        self.save_crop(crop, assigned_id, camera_id)
        return assigned_id

    def save_crop(self, crop, gid, camera_id):
        """
        Save the crop to disk for later review or training.

        Args:
            crop (np.ndarray): Cropped image to save.
            gid (int): Global ID assigned to this person.
            camera_id (str): Source camera ID.
        """
        filename = f"reid_data/{camera_id}_gid{gid}_frame{self.frame_id}.png"
        try:
            cv2.imwrite(filename, crop)
            print(f"[💾] Saved: {filename}")
        except Exception as e:
            print(f"[❌] Failed to save crop: {e}")
        self.frame_id += 1
