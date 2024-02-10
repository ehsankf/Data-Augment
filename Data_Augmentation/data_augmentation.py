import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import random
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
class data_augmentation():
    def __int__(self):
        pass
        # self.img = img
        # self.background = background

    def identity(self, image):
        return image
    def flip_horizi(self, image):
        flipped_image = cv2.flip(image, 1)  # 1 for horizontal flip
        return flipped_image

    def flip_verti(self, image):
            flipped_image = cv2.flip(image, 0)  # 1 for horizontal flip
            return flipped_image

    def rotate (self, image):
            angle = random.randint(0, 45)
            rows, cols, _ = image.shape
            center = (cols // 2, rows // 2)
            #angle = 30  # Rotation angle in degrees
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
            return rotated_image
    def random_crop(self, image):
        height, width = image.shape[:2]

        crop_width, crop_height = width//2, height//2

        if crop_width > width or crop_height > height:
            raise ValueError("Crop size is larger than image dimensions.")

        x = random.randint(0, width - crop_width)
        y = random.randint(0, height - crop_height)

        cropped_image = image[y:y+crop_height, x:x+crop_width]

        return cropped_image

    def random_translation(self, image):
        height, width = image.shape[:2]
        max_shift = random.randint(2, 200)
        # Generate random shift values within the specified range
        horizontal_shift = np.random.randint(-max_shift, max_shift)
        vertical_shift = np.random.randint(-max_shift, max_shift)

        # Create the translation matrix
        translation_matrix = np.float32([[1, 0, horizontal_shift], [0, 1, vertical_shift]])

        # Apply the translation to the image
        translated_image = cv2.warpAffine(image, translation_matrix, (width, height))

        return translated_image

    def scale_image(self, image, scale_percent=100):
        scale_percent = random.randint(75, 200)
        width = int(image.shape[1] * scale_percent / 100)
        scale_percent = random.randint(75, 200)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)

        scaled_image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)

        return scaled_image

    # def resize_image(self, image, new_width, new_height):
    #     resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # return resized_image

    def add_gaussian_noise(self, image, mean=0, std_dev=.5):
        noise = np.random.normal(mean, std_dev, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return noisy_image

    def adjust_brightness(self, image, value=20):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] + value, 0, 255)
        adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return adjusted_image

    def adjust_contrast(self, image, value=3):
        adjusted_image = np.clip(image * value, 0, 255).astype(np.uint8)
        return adjusted_image

    def color_jitter(self, image, hue_shift=10, saturation_shift=0.5, brightness_shift=20):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Hue adjustment
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180

        # Saturation adjustment
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * (1 + saturation_shift), 0, 255)

        # Brightness adjustment
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] + brightness_shift, 0, 255)

        jittered_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return jittered_image

    def shear_image(self, image, shear_factor=0.5):
        height, width = image.shape[:2]

        # Define the shear matrix
        shear_matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])

        # Apply the shear transformation to the image
        sheared_image = cv2.warpAffine(image, shear_matrix, (width, height))

        return sheared_image

    # def zoom_image(self, image, zoom_range):
    #     height, width = image.shape[:2]
    #
    #     # Calculate the center point of the image
    #     center_x = width // 2
    #     center_y = height // 2
    #
    #     # Generate a random zoom factor within the specified range
    #     zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])
    #
    #     # Calculate the new width and height after zooming
    #     new_width = int(width * zoom_factor)
    #     new_height = int(height * zoom_factor)
    #
    #     # Calculate the top-left corner coordinates for cropping the zoomed image
    #     top = center_y - new_height // 2
    #     left = center_x - new_width // 2
    #
    #     # Crop the zoomed image
    #     zoomed_image = image[top:top + new_height, left:left + new_width]
    #
    #     # Resize the zoomed image back to the original size
    #     zoomed_image = cv2.resize(zoomed_image, (width, height), interpolation=cv2.INTER_LINEAR)
    #
    #     return zoomed_image

    def elastic_transform(self, image):

        alpha = 200; sigma = 10
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        random_state = np.random.RandomState(None)

        # Generate random displacement fields
        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect") * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect") * alpha

        # Create the mesh grid
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

        # Apply the displacement fields to the mesh grid
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        # Map the image pixels to the displaced coordinates
        transformed_image = map_coordinates(image, indices, order=1, mode='reflect')
        transformed_image = transformed_image.reshape(shape)
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_GRAY2RGB)
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return transformed_image

    def perspective_transform(self, image):
        points = [(50, 50), (200, 50), (200, 200), (50, 200)]
        # Define the source and destination points for the perspective transform
        src_points = np.float32(points)
        dst_points = np.float32([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]])

        # Compute the perspective transform matrix
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply the perspective transform to the image
        transformed_image = cv2.warpPerspective(image, transform_matrix, (image.shape[1], image.shape[0]))

        return transformed_image

    def cutout(self, image, size=100, constant_value=None):
        height, width = image.shape[:2]

        if constant_value is None:
            # Generate random constant value within the image range
            constant_value = np.random.randint(0, 256)

        # Select random coordinates for the top-left corner of the cutout region
        top = np.random.randint(0, height - size)
        left = np.random.randint(0, width - size)

        # Replace the selected region with the constant value
        image[top:top + size, left:left + size] = constant_value

        return image

    def histogram_equalization(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)

        # Convert the equalized image back to BGR color space
        equalized_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

        return equalized_image

    def gaussian_blur(self, image, kernel_size=11, sigma=0):
        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

        return blurred_image

    # def motion_blur(self, image, kernel_size=15, angle=45):
    #     # Generate the motion blur kernel
    #     kernel = self.motion_blur_kernel(kernel_size, angle)
    #
    #     # Apply the motion blur kernel to the image
    #     blurred_image = cv2.filter2D(image, -1, kernel)
    #
    #     return blurred_image
    #
    # def motion_blur_kernel(self, kernel_size, angle):
    #     # Create a horizontal motion blur kernel
    #     kernel = np.zeros((kernel_size, kernel_size))
    #     center = kernel_size // 2
    #
    #     # Calculate the slope of the motion blur
    #     slope = np.tan(np.deg2rad(angle))
    #
    #     if slope <= 1:
    #         for i in range(kernel_size):
    #             offset = int(round(slope * (i - center)))
    #             kernel[i, center - offset] = 1.0 / kernel_size
    #     else:
    #         for i in range(kernel_size):
    #             offset = int(round((i - center) / slope))
    #             kernel[center - offset, i] = 1.0 / kernel_size
    #
    #     return kernel

    def random_skew(self, image, magnitude=0.2):
        height, width = image.shape[:2]

        # Define the skew range
        skew_range = np.random.uniform(-magnitude, magnitude)

        # Define the skew transformation matrix
        skew_matrix = np.array([[1, skew_range, 0], [0, 1, 0]])

        # Apply the skew transformation
        skewed_image = cv2.warpAffine(image, skew_matrix, (width, height), borderMode=cv2.BORDER_REFLECT)

        return skewed_image

    # def grid_mask(self,  image, grid_size=4, mask_ratio=0.5, fill_value=0):
    #     height, width = image.shape[:2]
    #
    #     # Calculate the size of each grid cell
    #     cell_height = height // grid_size
    #     cell_width = width // grid_size
    #
    #     # Create a blank mask image
    #     mask = np.ones_like(image) * fill_value
    #
    #     # Apply the grid mask
    #     for i in range(grid_size):
    #         for j in range(grid_size):
    #             # Calculate the top-left and bottom-right coordinates of the grid cell
    #             top = i * cell_height
    #             left = j * cell_width
    #             bottom = top + cell_height
    #             right = left + cell_width
    #
    #             # Check if the grid cell should be masked
    #             if np.random.uniform() < mask_ratio:
    #                 mask[top:bottom, left:right] = fill_value
    #
    #     # Apply the mask to the image
    #     masked_image = cv2.bitwise_and(image, mask)
    #
    #     return masked_image

    # def grid_distortion(self, image, grid_size, magnitude):
    #     height, width = image.shape[:2]
    #
    #     # Generate coordinates for the grid
    #     x = np.linspace(0, width, grid_size, endpoint=False).astype(int)
    #     y = np.linspace(0, height, grid_size, endpoint=False).astype(int)
    #
    #     # Apply random offsets to the grid points
    #     x += np.random.randint(-magnitude, magnitude, size=(grid_size,))
    #     y += np.random.randint(-magnitude, magnitude, size=(grid_size,))
    #
    #     # Create a meshgrid from the coordinates
    #     x_mesh, y_mesh = np.meshgrid(x, y)
    #
    #     # Reshape the meshgrid to a 2D array of points and convert to float32 type
    #     points = np.vstack((x_mesh.flatten(), y_mesh.flatten())).T.astype(np.float32)
    #
    #     # Create the destination points as the same grid coordinates
    #     dst_points = np.vstack((x_mesh.flatten(), y_mesh.flatten())).T.astype(np.float32)
    #
    #     # Compute the grid distortion using the remap function
    #     distorted_image = cv2.remap(image, points, dst_points, interpolation=cv2.INTER_LINEAR,
    #                                 borderMode=cv2.BORDER_REFLECT)
    #
    #     return distorted_image


def test():
    path = '/home/ehsan/PycharmProjects/SwinT_detectron2-main/background_edit/20210719165212822_1_confirmRGBRight_3c.png'

    aug = data_augmentation()
    functions = [func for func in dir(aug) if callable(getattr(aug, func)) and not func.startswith('__')]
    selected_names = random.sample(functions, 4)
    image = cv2.imread(path)

    for function_ in selected_names:
        print(function_)
        image = getattr(aug, function_)(image)
        plt.imshow(image)
        plt.show()

    alpha = 200  # Elastic distortion intensity
    sigma = 10  # Smoothing factor
    grid_size = 1 # Size of the grid
    magnitude = 20  # Magnitude of distortion

    # Apply grid distortion
    im = aug.resize_image(image, 500, 500)
    plt.imshow(im)

    plt.show()

    # Convert the image to grayscale
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply elastic transformations
    #points = [(50, 50), (200, 50), (200, 200), (50, 200)]


if __name__ =="main":
    test()



