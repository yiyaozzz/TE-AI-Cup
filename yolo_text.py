import os


def create_dataset_file(image_dir, output_file):
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(root, file)
                    f.write(path + '\n')


create_dataset_file('datasets/train', 'datasets')
create_dataset_file('datasets/val', 'datasets')
