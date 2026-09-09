import os
import glob
from src.data.emidec_loader import EMIDECCase
from src.visualizer.viewer import DatasetViewer

def main():
    train_dir = "emidec_dataset/train"
    case_dirs = sorted([d for d in glob.glob(f"{train_dir}/Case_*") if os.path.isdir(d)])
    if not case_dirs:
        print(f"Error: No cases found in {train_dir}")
        return
    
    cases = [EMIDECCase(d) for d in case_dirs]
    print(f"Loaded {len(cases)} cases. Launching viewer...")
    viewer = DatasetViewer(cases)
    viewer.show()

if __name__ == "__main__":
    main()
