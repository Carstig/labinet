import xml.etree.ElementTree as ET


def get_normalized_coordinates(box, image_size):
    '''
    return box coordinates resized to image_size
    left, right, top, bottom aka xmin, xmax, ymin, ymax
    '''
    (ymin, xmin, ymax, xmax) = box
    im_width = image_size[0]
    im_height = image_size[1]
    (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height))
    box = { 'xmin': left, 'xmax': right, 'ymin': top, 'ymax': bottom }
    return box


def box_to_labelimg_xml(filename, image_size, boxes, label='dog', imagepath=None, ):
    '''
    The result from object detection converted into an XML document matching 
    the format of the labelImg tool = PascalVOC
    @param filename is image name
    @param image_size
    @param boxes list of boxes from object_detection // boxes_to_use
    @label the label to use for _every_ box (might change this when I use more than 1 label)
    '''
    root = ET.Element('annotation')
    folder = ET.SubElement(root, 'folder')
    folder.text = 'images'
    fname = ET.SubElement(root, 'filename')
    fname.text = filename
    pathname = ET.SubElement(root, 'path')
    if imagepath is not None:
        pathname.text = imagepath
    else:
        pathname.text = ""
    source = ET.SubElement(root, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'
    
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(image_size[0])
    height = ET.SubElement(size, 'height')
    height.text = str(image_size[1])
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    
    segmented = ET.SubElement(root, 'segmented')
    segmented.text = '0'
    
    ### 1 box. each box becomes 'object'
    for box in boxes:
        obj = ET.SubElement(root, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = label
        pose = ET.SubElement(obj, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = '0'
        bndbox = ET.SubElement(obj, 'bndbox')
        xminx = ET.SubElement(bndbox, 'xmin')
        xminx.text = str(box['xmin'])
        xmaxx = ET.SubElement(bndbox, 'xmax')
        xmaxx.text = str(box['xmax'])
        yminx = ET.SubElement(bndbox, 'ymin')
        yminx.text = str(box['ymin'])
        ymaxx = ET.SubElement(bndbox, 'ymax')
        ymaxx.text = str(box['ymax'])

    tree = ET.ElementTree(root)
    return tree
    