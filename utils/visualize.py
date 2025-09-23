import json
import numpy as np
from PIL import Image, ImageDraw

def read_cocoa_json(file_path):
    """
    Đọc file COCOA JSON
    
    Args:
        file_path: Đường dẫn tới file JSON
    
    Returns:
        dict: Dữ liệu từ file JSON
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def combine_masks_horizontal(visible_path, amodal_path, output_path):
    """
    Nối 2 PNG masks thành 1 ảnh ngang
    
    Args:
        visible_path: Đường dẫn visible mask
        amodal_path: Đường dẫn amodal mask
        output_path: Đường dẫn ảnh kết hợp
    """
    # Mở 2 ảnh
    visible_img = Image.open(visible_path)
    amodal_img = Image.open(amodal_path)
    
    # Lấy kích thước
    width, height = visible_img.size
    
    # Tạo ảnh mới có chiều rộng gấp đôi
    combined_img = Image.new('L', (width * 2, height), 0)
    
    # Dán visible mask bên trái
    combined_img.paste(visible_img, (0, 0))
    
    # Dán amodal mask bên phải  
    combined_img.paste(amodal_img, (width, 0))
    
    # Lưu ảnh kết hợp
    combined_img.save(output_path)
    print(f"Saved combined mask: {output_path}")

def polygon_to_png(polygon, image_width, image_height, output_path):
    """
    Chuyển polygon thành PNG mask
    
    Args:
        polygon: List tọa độ [x1, y1, x2, y2, ...]
        image_width: Chiều rộng ảnh
        image_height: Chiều cao ảnh  
        output_path: Đường dẫn lưu PNG
    """
    # Tạo ảnh trống (đen)
    img = Image.new('L', (image_width, image_height), 0)
    draw = ImageDraw.Draw(img)
    
    # Chuyển polygon thành list of tuples
    if len(polygon) >= 6:  # Cần ít nhất 3 điểm
        points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
        draw.polygon(points, fill=255)  # Vẽ polygon màu trắng (255)
    
    # Lưu PNG
    img.save(output_path)
    print(f"Saved mask: {output_path}")

# Cách sử dụng
if __name__ == "__main__":
    # Đọc file
    file_path = "train/cocoa_format_annotations.json"
    cocoa_data = read_cocoa_json(file_path)
    
    # Truy cập các phần dữ liệu
    images = cocoa_data.get('images', [])
    annotations = cocoa_data.get('annotations', [])
    categories = cocoa_data.get('categories', [])
    
    # In thông tin cơ bản
    print(f"Images: {len(images)}")
    print(f"Annotations: {len(annotations)}")
    print(f"Categories: {len(categories)}")
    
    # Lấy kích thước ảnh (giả sử từ image đầu tiên)
    if images:
        img_width = images[0]['width']
        img_height = images[0]['height']
    else:
        img_width, img_height = 512, 512  # Default size
    
    # Xử lý annotations
    if annotations:
        for annotation in annotations:
            if annotation['image_id'] > 0:
                break
            
            annotation_id = annotation['id']
            
            # Decode amodal mask
            if 'amodal_segm' in annotation and annotation['amodal_segm']:
                amodal_polygon = annotation['amodal_segm'][0]
                amodal_output = f"amodal_mask_{annotation_id}.png"
                polygon_to_png(amodal_polygon, img_width, img_height, amodal_output)
            
            # Decode visible mask  
            if 'visible_segm' in annotation and annotation['visible_segm']:
                visible_polygon = annotation['visible_segm'][0]
                visible_output = f"visible_mask_{annotation_id}.png"
                polygon_to_png(visible_polygon, img_width, img_height, visible_output)
                
                # Nối 2 masks thành 1 ảnh ngang
                if 'amodal_segm' in annotation and annotation['amodal_segm']:
                    combined_output = f"combined_mask_{annotation_id}.png"
                    combine_masks_horizontal(visible_output, amodal_output, combined_output)