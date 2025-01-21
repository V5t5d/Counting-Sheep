import cv2 as cv
import numpy as np

def detecting_area(frame):
    """
    Defines the region of interest (ROI) on the frame using a polygonal mask.
    Only the area within the polygon is considered for further processing.
    """
    h, w, _ = frame.shape  # Get the height and width of the frame
    color = (128, 128, 128)  # Define a color for the mask (not used in this function)

    # Define the vertices of the polygon for the ROI
    # These points form a polygon that outlines the area where detection will occur
    pts = np.array([[0, 0], [0, h], [w, h], [w, 0]])

    # Create a black mask of the same size as the frame
    mask = np.zeros_like(frame)
    
    # Fill the polygon defined by `pts` with white (1,1,1) on the mask
    cv.fillPoly(mask, [pts], (1, 1, 1))

    # Apply the mask to the frame: only the area within the polygon will remain visible
    img_masked = frame * mask

    # Uncomment the following lines to visualize the masked image
    # cv.imshow("test", img_masked)
    # cv.waitKey(0)

    return img_masked  # Return the masked frame => (ROI) is limited

def detecting_area_unlimited(frame):
    return frame  # Return the unmasked frame => (ROI) is not limited


def draw_bounds(frame):
    h, w, _ = frame.shape  # Get the height and width of the frame
    color = (0, 255, 255)  # Define a color for the boundary lines (yellow)

    # Define a polygon that covers the entire frame
    pts = np.array([[0, 0], [0, h], [w, h], [w, 0]])

    # Draw the boundaries of the rectangle
    frame = cv.line(frame, (0, 0), (0, h), color, 3)  # Left edge
    frame = cv.line(frame, (0, h), (w, h), color, 3)  # Bottom edge
    frame = cv.line(frame, (w, h), (w, 0), color, 3)  # Right edge
    frame = cv.line(frame, (w, 0), (0, 0), color, 3)  # Top edge

    # Highlight the entire frame (optional)
    color = (221, 218, 250)  # Light purple color
    mask = np.zeros((frame.shape[0], frame.shape[1]))
    cv.fillPoly(mask, [pts], (1, 1, 1))
    temp = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    temp[mask == 1] = color
    frame[mask == 1] = 0.7 * frame[mask == 1] + 0.3 * temp[mask == 1]

    return frame