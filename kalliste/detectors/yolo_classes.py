"""Available detection classes for YOLOv11x."""

#TODO: Fix these class numbers

YOLO_CLASSES = {
    # People and faces
    'person': 0,
    
#     # Animals
#     'bird': 14,
#     'cat': 15,
#     'dog': 16,
#     'horse': 17,
#     'sheep': 18,
#     'cow': 19,
#     'elephant': 20,
#     'bear': 21,
#     'zebra': 22,
#     'giraffe': 23,
    
#     # Vehicles
#     'bicycle': 1,
#     'car': 2,
#     'motorcycle': 3,
#     'airplane': 4,
#     'bus': 5,
#     'train': 6,
#     'truck': 7,
#     'boat': 8,
    
#     # Common objects
#     'traffic light': 9,
#     'fire hydrant': 10,
#     'stop sign': 11,
#     'parking meter': 12,
#     'bench': 13,
    
#     # Add more classes as needed...
}

# # Groups of related classes
CLASS_GROUPS = {
#     'animals': {'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'},
#     'vehicles': {'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'},
#     'street': {'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench'},
} 


# Actual coco classes:

#  0: person
#   1: bicycle
#   2: car
#   3: motorcycle
#   4: airplane
#   5: bus
#   6: train
#   7: truck
#   8: boat
#   9: traffic light
#   10: fire hydrant
#   11: stop sign
#   12: parking meter
#   13: bench
#   14: bird
#   15: cat
#   16: dog
#   17: horse
#   18: sheep
#   19: cow
#   20: elephant
#   21: bear
#   22: zebra
#   23: giraffe
#   24: backpack
#   25: umbrella
#   26: handbag
#   27: tie
#   28: suitcase
#   29: frisbee
#   30: skis
#   31: snowboard
#   32: sports ball
#   33: kite
#   34: baseball bat
#   35: baseball glove
#   36: skateboard
#   37: surfboard
#   38: tennis racket
#   39: bottle
#   40: wine glass
#   41: cup
#   42: fork
#   43: knife
#   44: spoon
#   45: bowl
#   46: banana
#   47: apple
#   48: sandwich
#   49: orange
#   50: broccoli
#   51: carrot
#   52: hot dog
#   53: pizza
#   54: donut
#   55: cake
#   56: chair
#   57: couch
#   58: potted plant
#   59: bed
#   60: dining table
#   61: toilet
#   62: tv
#   63: laptop
#   64: mouse
#   65: remote
#   66: keyboard
#   67: cell phone
#   68: microwave
#   69: oven
#   70: toaster
#   71: sink
#   72: refrigerator
#   73: book
#   74: clock
#   75: vase
#   76: scissors
#   77: teddy bear
#   78: hair drier
#   79: toothbrush