"""Available detection classes for YOLOv11x."""

YOLO_CLASSES = {
    # People and faces
    'person': 0,
    'face': 1,
    
    # Animals
    'bird': 14,
    'cat': 15,
    'dog': 16,
    'horse': 17,
    'sheep': 18,
    'cow': 19,
    'elephant': 20,
    'bear': 21,
    'zebra': 22,
    'giraffe': 23,
    
    # Vehicles
    'bicycle': 1,
    'car': 2,
    'motorcycle': 3,
    'airplane': 4,
    'bus': 5,
    'train': 6,
    'truck': 7,
    'boat': 8,
    
    # Common objects
    'traffic light': 9,
    'fire hydrant': 10,
    'stop sign': 11,
    'parking meter': 12,
    'bench': 13,
    
    # Add more classes as needed...
}

# Groups of related classes
CLASS_GROUPS = {
    'animals': {'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'},
    'vehicles': {'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'},
    'street': {'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench'},
} 