from Model import modelInter
import cv2
import numpy as np
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import segmentation_models_pytorch_local as smp

def avi_to_numpy(video_path):
    """
    Reads an .avi video file and converts each frame into a NumPy array.

    Args:
        video_path (str): Path to the video file.

    Returns:
        numpy.ndarray: NumPy array of shape (num_frames, 3, 512, 512).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize frame to (512, 512)
        frame_resized = cv2.resize(frame_rgb, (512, 512))

        # Convert to numpy array of shape (3, 512, 512)
        frame_np = np.transpose(frame_resized, (2, 0, 1))  # (512, 512, 3) -> (3, 512, 512)

        # Append to the list of frames
        frames.append(frame_np)

    cap.release()

    # Stack frames along a new axis to create a numpy array of shape (num_frames, 3, 512, 512)
    frames_np = np.stack(frames)

    return frames_np

frames = avi_to_numpy("test_video.avi")

# Initializing models
MDL = modelInter()
MDL.load()


print(frames[3].shape)
frames = torch.from_numpy(frames)
for i in range(frames.shape[0]):
    cls_logit, seg, seg_flag = MDL.predict(frames[i], 1)

    if(seg_flag):
        print("HERE")
        fig, ax = plt.subplots(2,3)

        print(frames.shape,seg.shape)

        # Display the frame in the first row, second column
        ax[0][0].imshow(frames[i][0])
        ax[0][1].imshow(frames[i][1])
        ax[0][2].imshow(frames[i][2])


        im = ax[1][0].imshow(seg, cmap="gray")

        # Attach the colorbar to the specific subplot with the segmentation image
        fig.colorbar(im, ax=ax[1][0])

        # Turn into pciture for task 3
        # Run RunHere.py to get measurments

        # Create an empty RGB image
        rgb_image = np.zeros((512, 512, 3), dtype=np.uint8)
        # Assign red color to value 1
        rgb_image[seg == 1] = [0, 0, 255]  # Red
        # Assign green color to value 2
        rgb_image[seg == 2] = [0, 255, 0]  # Green

        # Save the image
        cv2.imwrite('Task3/segmentation_result.png', rgb_image)

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()