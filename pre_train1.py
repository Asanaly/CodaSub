import os
import cv2


def extract_frames(video_path, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit if the video has ended

        # Save every second frame (0, 2, 4, ...)
        if frame_count % 2 == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()  # Release the video capture object


def process_videos(input_dir, output_dir):
    # Process negative samples (avi files directly in neg folder)
    neg_dir = os.path.join(input_dir, "neg")
    for filename in os.listdir(neg_dir):
        if filename.endswith(".avi"):
            video_path = os.path.join(neg_dir, filename)
            neg_output_dir = os.path.join(output_dir, "neg")
            extract_frames(video_path, neg_output_dir)

    # Process positive samples (avi files inside folders in pos folder)
    pos_dir = os.path.join(input_dir, "pos")
    for foldername in os.listdir(pos_dir):
        folder_path = os.path.join(pos_dir, foldername)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith(".avi"):
                    video_path = os.path.join(folder_path, filename)
                    pos_output_dir = os.path.join(output_dir, "pos")
                    extract_frames(video_path, pos_output_dir)


if __name__ == "__main__":
    input_directory = "./dataset/train"  # Replace with your input folder path
    output_directory = "./dataset2/train"  # Replace with your desired output folder path
    process_videos(input_directory, output_directory)

    input_directory = "./dataset/val"  # Replace with your input folder path
    output_directory = "./dataset2/val"  # Replace with your desired output folder path
    process_videos(input_directory, output_directory)
