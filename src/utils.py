import os
import random
import shutil

def split_train_test(data_dir, output_dir, test_ratio=0.2, seed=42):
    random.seed(seed)
    classes = os.listdir(data_dir)

    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        images = os.listdir(cls_path)
        random.shuffle(images)

        split_idx = int(len(images) * (1 - test_ratio))
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        for phase, img_list in zip(["train", "test"], [train_images, test_images]):
            dest = os.path.join(output_dir, phase, cls)
            os.makedirs(dest, exist_ok=True)
            for img in img_list:
                shutil.copy(os.path.join(cls_path, img), os.path.join(dest, img))

    print("Dataset split into train/test successfully!")
