import numpy as np
import cv2
from PIL import Image


# def save_bound_box_from_annotations(input_path, output_path):
#     pil_img = Image.open(input_path)
#     pil_img = pil_img.convert("L")
#     img = np.array(pil_img) * 255
#     contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     bounding_boxes = []
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         bounding_boxes.append((x, y, x + w, y + h))
#     output_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     for box in bounding_boxes:
#         cv2.rectangle(output_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
#     cv2.imwrite(output_path, output_image)
#     return (x, y, x + w, y + h)

def save_bound_box_from_annotations(input_path, output_image_path, confidence):
    pil_img = Image.open(input_path)
    pil_img = pil_img.convert("L")
    img = np.array(pil_img) * 255
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, x + w, y + h))  # Original bounding box
    output_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for box in bounding_boxes:
        cv2.rectangle(output_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    
    # Create a separate row for the confidence text
    confidence_text = f'Confidence: {confidence}'
    text_height = 20  # Height of the text row
    text_width = output_image.shape[1]  # Width of the image
    text_image = np.ones((text_height, text_width, 3), np.uint8) * 255  # Create a white row
    cv2.putText(text_image, confidence_text, (10, text_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Add text
    
    # Combine the image and the confidence row
    combined_image = np.vstack((output_image, text_image))
    
    cv2.imwrite(output_image_path, combined_image)
    
    return bounding_boxes
        
if __name__ == "__main__":
    img_path = r'/home/ma22resch11003/wce_code_base/results/validation/WCE_10054.png'
    save_bound_box_from_annotations(img_path, "/home/ma22resch11003/wce_code_base/results/validation/WCE_10054_bb.png", 0.928)


# pil_img = Image.open(img_path)
# pil_img = pil_img.convert("L")
# # pil_img.save("pil.png")
# img_array = np.array(pil_img) * 255
# pil_img = Image.fromarray(img_array)
# # pil_img.save("pil.png")
# # print(np.array(pil_img).shape)

# img = np.array(pil_img)
# # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# bounding_boxes = []
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     bounding_boxes.append((x, y, x + w, y + h))  # (x_min, y_min, x_max, y_max)

# output_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# for box in bounding_boxes:
#     cv2.rectangle(output_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    
# cv2.imwrite('output_image.png', output_image)

# # cv2.imshow('Bounding Boxes', output_image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

