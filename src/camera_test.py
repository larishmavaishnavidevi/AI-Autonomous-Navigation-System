# src/camera_test.py
import cv2

def test_camera():
    """Activates the webcam and captures real-time video."""
    print("Initializing camera... Press 'q' on your keyboard to exit.")
    
    # Open the default camera (0 is usually your laptop's built-in webcam)
    cap = cv2.VideoCapture(0)
    
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open the camera. Please check your permissions.")
        return

    while True:
        # Read the video frame by frame
        ret, frame = cap.read()
        
        # If the frame was not read correctly, break the loop
        if not ret:
            print("Error: Failed to grab a frame.")
            break
            
        # Display the resulting frame in a window named 'Live Camera Feed'
        cv2.imshow("Live Camera Feed", frame)
        
        # Wait for 1 millisecond for a key press. If 'q' is pressed, exit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting camera feed...")
            break
            
    # When everything is done, release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()