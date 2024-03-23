import os
import random
import csv

def collect_img_and_label_for_one_dataset():
        """Collects image and label lists.

        Args:
            dataset_name (str): A list containing one dataset information. e.g., 'FF-F2F'

        Returns:
            list: A list of image paths.
            list: A list of labels.
        
        Raises:
            ValueError: If image paths or labels are not found.
            NotImplementedError: If the dataset is not implemented yet.
        """
    
        # Initialize the label and frame path lists
        label_list = []
        frame_path_list = []
        root_dir = ""
        csv_file = "/Users/devikapillai/Desktop/DeFake/datasets/copied_image_paths.csv"

        with open(csv_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                image_path = os.path.join(root_dir, line.strip())
                label = 1 if '/fake/' in line else 0
                frame_path_list.append(image_path)
                label_list.append(label)

        # Shuffle the label and frame path lists in the same order
        shuffled = list(zip(label_list, frame_path_list))
        random.shuffle(shuffled)
        label_list, frame_path_list = zip(*shuffled)

        print(frame_path_list[430:434])
        print(label_list[430:434])

        
        
        return frame_path_list, label_list


if __name__ == "__main__":
    collect_img_and_label_for_one_dataset()