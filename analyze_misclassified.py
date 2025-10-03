import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import shutil

# Path to predictions CSV
csv_path = "predictions.csv"

# Load predictions
df = pd.read_csv(csv_path)

# Filter misclassified images
misclassified = df[df['True_Label'] != df['Predicted_Label']]
print(f"Total misclassified images: {len(misclassified)}")
print(misclassified.head())

# Folder containing test images
test_folder = "./test"

# Scroll through all misclassified images
for idx, row in misclassified.iterrows():
    img_path = os.path.join(test_folder, row['True_Label'], row['Image_Name'])
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        continue
    img = mpimg.imread(img_path)
    plt.imshow(img, cmap='gray')
    plt.title(f"True: {row['True_Label']} | Predicted: {row['Predicted_Label']}")
    plt.axis('off')
    plt.show()
    
    # Optional: pause for user input to continue
    input("Press Enter to see next image...")

# Save all misclassified images to a folder
output_folder = "./misclassified"
os.makedirs(output_folder, exist_ok=True)

for idx, row in misclassified.iterrows():
    src_path = os.path.join(test_folder, row['True_Label'], row['Image_Name'])
    dst_path = os.path.join(output_folder, f"{row['True_Label']}_{row['Predicted_Label']}_{row['Image_Name']}")
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)

# Save misclassified summary CSV
misclassified.to_csv("misclassified_summary.csv", index=False)

print(f"\n✅ All misclassified images saved to '{output_folder}'")
print("✅ Summary CSV saved as 'misclassified_summary.csv'")
