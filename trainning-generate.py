from PIL import Image
import random
import os


def generate_images_with_labels(background_path, object_paths, output_dir, num_images=10, num_objects=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    background = Image.open(background_path).convert('RGBA')
    objects = [Image.open(path).convert('RGBA') for path in object_paths]

    for img_index in range(1, num_images + 1):
        temp_background = background.copy()
        labels = []

        for _ in range(num_objects):
            obj_index = random.randint(0, len(objects) - 1)
            obj = objects[obj_index].rotate(random.randint(0, 360), expand=True)
            obj_width, obj_height = obj.size

            print(f"Object {obj_index} rotated size: {obj_width} x {obj_height}")  # Debug: Output size after rotation

            if obj_width < temp_background.width and obj_height < temp_background.height:
                x = random.randint(0, temp_background.width - obj_width)
                y = random.randint(0, temp_background.height - obj_height)

                x_center = (x + obj_width / 2) / temp_background.width
                y_center = (y + obj_height / 2) / temp_background.height
                width = obj_width / temp_background.width
                height = obj_height / temp_background.height
                labels.append(f"{obj_index} {x_center} {y_center} {width} {height}")
                mask = obj.split()[-1]
                temp_background.paste(obj, (x, y), mask)
                print(f"Object {obj_index} pasted at ({x}, {y}).")  # Debug: Confirm object pasting
            else:
                print(f"Object {obj_index} not pasted due to size.")  # Debug: Object skipped due to size

        image_filename = f"{img_index}.png"
        label_filename = f"{img_index}.txt"

        temp_background.save(os.path.join(output_dir, image_filename))
        with open(os.path.join(output_dir, label_filename), 'w') as f:
            for label in labels:
                f.write(label + '\n')


# Example usage
generate_images_with_labels(
    'assets/background.png',
    ['assets/bullet.png', 'assets/bullet-enemy.png', 'assets/charactor.png'],
    'output_directory_5.27（2）',
    num_images=899,  # 生成50张图像
    num_objects=3
)
