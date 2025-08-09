import os
import shutil
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

class DataPreprocessor:
    def __init__(self, dataset_root="dataset/train"):
        self.dataset_root = dataset_root
        self.class_names = [
            '01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb',
            '06_index', '07_ok', '08_palm_moved', '09_c', '10_down'
        ]
    
    def analyze_dataset(self):
        """Analyze the dataset structure and statistics"""
        print("=" * 60)
        print("DATASET ANALYSIS")
        print("=" * 60)
        
        total_images = 0
        class_counts = {}
        
        # Check if dataset exists
        if not os.path.exists(self.dataset_root):
            print(f"❌ Dataset root '{self.dataset_root}' not found!")
            return
        
        # Get all subject folders (00, 01, 02, ...)
        subject_folders = sorted([f for f in os.listdir(self.dataset_root) 
                                if os.path.isdir(os.path.join(self.dataset_root, f))])
        
        print(f"📁 Found {len(subject_folders)} subject folders: {subject_folders}")
        print()
        
        for class_name in self.class_names:
            class_counts[class_name] = 0
        
        # Analyze each subject folder
        for subject_folder in subject_folders:
            subject_path = os.path.join(self.dataset_root, subject_folder)
            print(f"📊 Analyzing subject {subject_folder}:")
            
            for class_name in self.class_names:
                class_path = os.path.join(subject_path, class_name)
                if os.path.exists(class_path):
                    images = [f for f in os.listdir(class_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    count = len(images)
                    class_counts[class_name] += count
                    total_images += count
                    print(f"  {class_name}: {count} images")
                else:
                    print(f"  {class_name}: ❌ MISSING")
            print()
        
        print("=" * 60)
        print("OVERALL STATISTICS")
        print("=" * 60)
        print(f"Total images: {total_images}")
        print(f"Total classes: {len(self.class_names)}")
        print(f"Average images per class: {total_images / len(self.class_names):.1f}")
        print()
        
        print("Class distribution:")
        for class_name, count in class_counts.items():
            percentage = (count / total_images) * 100 if total_images > 0 else 0
            print(f"  {class_name}: {count} images ({percentage:.1f}%)")
        
        return class_counts, total_images
    
    def create_unified_dataset(self, output_dir="dataset/unified"):
        """Create a unified dataset from all subject folders"""
        print("\n" + "=" * 60)
        print("CREATING UNIFIED DATASET")
        print("=" * 60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create class directories
        for class_name in self.class_names:
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
        
        # Get all subject folders
        subject_folders = sorted([f for f in os.listdir(self.dataset_root) 
                                if os.path.isdir(os.path.join(self.dataset_root, f))])
        
        total_copied = 0
        
        for subject_folder in subject_folders:
            subject_path = os.path.join(self.dataset_root, subject_folder)
            print(f"Processing subject {subject_folder}...")
            
            for class_name in self.class_names:
                source_class_path = os.path.join(subject_path, class_name)
                target_class_path = os.path.join(output_dir, class_name)
                
                if os.path.exists(source_class_path):
                    images = [f for f in os.listdir(source_class_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    
                    for image in images:
                        source_file = os.path.join(source_class_path, image)
                        # Create unique filename with subject prefix
                        new_filename = f"{subject_folder}_{image}"
                        target_file = os.path.join(target_class_path, new_filename)
                        
                        shutil.copy2(source_file, target_file)
                        total_copied += 1
        
        print(f"✅ Copied {total_copied} images to unified dataset")
        print(f"📁 Unified dataset created at: {output_dir}")
        
        return output_dir
    
    def visualize_samples(self, num_samples=2):
        """Visualize sample images from each class"""
        print("\n" + "=" * 60)
        print("VISUALIZING SAMPLE IMAGES")
        print("=" * 60)
        
        fig, axes = plt.subplots(len(self.class_names), num_samples, 
                               figsize=(num_samples * 3, len(self.class_names) * 2))
        
        if len(self.class_names) == 1:
            axes = [axes]
        
        for i, class_name in enumerate(self.class_names):
            # Find first subject folder that has this class
            sample_images = []
            
            subject_folders = sorted([f for f in os.listdir(self.dataset_root) 
                                    if os.path.isdir(os.path.join(self.dataset_root, f))])
            
            for subject_folder in subject_folders:
                class_path = os.path.join(self.dataset_root, subject_folder, class_name)
                if os.path.exists(class_path):
                    images = [f for f in os.listdir(class_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    for img_file in images[:num_samples - len(sample_images)]:
                        img_path = os.path.join(class_path, img_file)
                        sample_images.append(img_path)
                        
                        if len(sample_images) >= num_samples:
                            break
                
                if len(sample_images) >= num_samples:
                    break
            
            # Display samples
            for j in range(num_samples):
                if j < len(sample_images):
                    img = cv2.imread(sample_images[j])
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        axes[i][j].imshow(img)
                        axes[i][j].set_title(f"{class_name}")
                    else:
                        axes[i][j].text(0.5, 0.5, 'No Image', 
                                       ha='center', va='center')
                else:
                    axes[i][j].text(0.5, 0.5, 'No Image', 
                                   ha='center', va='center')
                
                axes[i][j].axis('off')
        
        plt.tight_layout()
        plt.savefig('dataset_samples.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def check_image_quality(self):
        """Check for corrupted or low-quality images"""
        print("\n" + "=" * 60)
        print("CHECKING IMAGE QUALITY")
        print("=" * 60)
        
        corrupted_images = []
        small_images = []
        
        subject_folders = sorted([f for f in os.listdir(self.dataset_root) 
                                if os.path.isdir(os.path.join(self.dataset_root, f))])
        
        for subject_folder in subject_folders:
            subject_path = os.path.join(self.dataset_root, subject_folder)
            
            for class_name in self.class_names:
                class_path = os.path.join(subject_path, class_name)
                
                if os.path.exists(class_path):
                    images = [f for f in os.listdir(class_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    
                    for image_file in images:
                        image_path = os.path.join(class_path, image_file)
                        
                        try:
                            img = cv2.imread(image_path)
                            if img is None:
                                corrupted_images.append(image_path)
                            else:
                                height, width = img.shape[:2]
                                if height < 50 or width < 50:
                                    small_images.append((image_path, height, width))
                        except Exception as e:
                            corrupted_images.append(image_path)
        
        print(f"🔍 Quality check results:")
        print(f"  Corrupted images: {len(corrupted_images)}")
        print(f"  Small images (< 50px): {len(small_images)}")
        
        if corrupted_images:
            print("\n❌ Corrupted images found:")
            for img_path in corrupted_images[:5]:  # Show first 5
                print(f"  {img_path}")
            if len(corrupted_images) > 5:
                print(f"  ... and {len(corrupted_images) - 5} more")
        
        if small_images:
            print("\n⚠️  Small images found:")
            for img_path, h, w in small_images[:5]:  # Show first 5
                print(f"  {img_path} ({w}x{h})")
            if len(small_images) > 5:
                print(f"  ... and {len(small_images) - 5} more")
        
        return corrupted_images, small_images

def main():
    """Main preprocessing pipeline"""
    preprocessor = DataPreprocessor()
    
    # Analyze dataset
    class_counts, total_images = preprocessor.analyze_dataset()
    
    if total_images == 0:
        print("No images found in dataset. Please check your dataset structure.")
        return
    
    # Check image quality
    corrupted, small = preprocessor.check_image_quality()
    
    # Visualize samples
    preprocessor.visualize_samples(num_samples=3)
    
    # Ask if user wants to create unified dataset
    response = input("\nDo you want to create a unified dataset? (y/n): ")
    if response.lower() == 'y':
        unified_path = preprocessor.create_unified_dataset()
        print(f"\n✅ Dataset preprocessing completed!")
        print(f"📁 Unified dataset available at: {unified_path}")
        print("You can now use this unified dataset for training.")
    
    print("\n🎯 Dataset is ready for training!")

if __name__ == "__main__":
    main()