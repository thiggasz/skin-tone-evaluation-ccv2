import cv2
import numpy as np

def get_skin_pixels(img_face, img_mask, erosion_iters = 2):
    """Given a image of segmentated skin, returns the median BGR color and a skin pixels array"""
    
    if len(img_mask.shape) == 3:
        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    
    # Erodes the mask to eliminate darker skin pixels on the face edge
    kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(img_mask, kernel, iterations=erosion_iters)
    
    # Obtain the skin pixels based on the mask
    is_skin = eroded_mask > 127
    skin_pixels_bgr = img_face[is_skin]
    
    if len(skin_pixels_bgr) < 10:
        return skin_pixels_bgr
        
    # Convert the skin pixels to CIELAB
    skin_pixels_bgr_reshaped = skin_pixels_bgr.reshape(-1, 1, 3).astype(np.uint8)
    skin_pixels_lab = cv2.cvtColor(skin_pixels_bgr_reshaped, cv2.COLOR_BGR2LAB)
    
    luminance = skin_pixels_lab[:, 0, 0]
    
    # Disregard pixels with extreme luminaces
    L_LOWER = 50  
    L_UPPER = 215 

    valid_mask = (luminance > L_LOWER) & (luminance < L_UPPER)
    valid_luminance = luminance[valid_mask]
    
    # Tries to eliminate luminance variation 
    if len(valid_luminance) > 10:
        Q1 = np.percentile(valid_luminance, 25)
        Q3 = np.percentile(valid_luminance, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        final_mask = valid_mask & (luminance >= lower_bound) & (luminance <= upper_bound)
    else:
        final_mask = valid_mask

    selected_skin_pixels = skin_pixels_bgr[final_mask]

    if len(selected_skin_pixels) < 10:
        return None
    
    return selected_skin_pixels