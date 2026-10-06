import os
import shutil
import time
import subprocess

def main():
    best_pt_src = "runs/detect/weld_train/weights/best.pt"
    
    # Wait until best.pt is created (polling just in case, but training should be done soon)
    print("Waiting for training to complete and best.pt to be generated...")
    while not os.path.exists(best_pt_src):
        time.sleep(5)
        
    print("Training complete! Copying best.pt to root directory...")
    shutil.copy(best_pt_src, "./best.pt")
    
    print("\nRunning metrics script...")
    subprocess.run(["python3", "print_metrics.py"])
    
if __name__ == "__main__":
    main()
