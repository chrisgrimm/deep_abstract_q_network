import cv2
import os
import numpy as np

base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../screenshots')
screenshot = lambda name: cv2.imread(os.path.join(base_path, name+'.png'))

def get_agent_position(screen):
    agent_conv = screenshot('Room(6,3)')[29:29+9, 79:79+5,0]
    detector_image = cv2.matchTemplate(screen, agent_conv, cv2.TM_SQDIFF_NORMED)
    return np.array(np.unravel_index(np.argmin(detector_image), detector_image.shape)) / 84.


print get_agent_position(screenshot('Room(6,3)')[:, :, 0])

# definition of montezumas nodes

classifier_5_1 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(58, 45, 18, 17, screenshot('Room(5,1)Key(True)Door(True)Door(True)')))
room_5_1 = ah.ClassifierNode((5,1))\
    .set_classifier(classifier_5_1)\
    .add_state_check(ah.ImageCheck(6, 39, 5, 7, screenshot('Room(5,1)Key(False)Door(True)Door(True)')), '(5,1)Key', False)\
    .add_state_check(ah.ImageCheck(71, 21, 3, 15, screenshot('Room(5,1)Key(False)Door(True)Door(False)')), '(5,1)RightDoor', False)\
    .add_state_check(ah.ImageCheck(10, 20, 3, 17, screenshot('Room(5,1)Key(False)Door(False)Door(True)')), '(5,1)LeftDoor', False)

classifier_6_1 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(32, 39, 20, 15, screenshot('Room(6,1)')))\
    .add_check(ah.ImageCheck(81, 16, 3, 14, screenshot('Room(6,1)')))
room_6_1 = ah.ClassifierNode((6,1))\
    .set_classifier(classifier_6_1)

classifier_4_1 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(31, 39, 23, 15, screenshot('Room(4,1)')))\
    .add_check(ah.ImageCheck(0, 15, 3, 14, screenshot('Room(4,1)')))
room_4_1 = ah.ClassifierNode((4,1))\
    .set_classifier(classifier_4_1)\
    #.add_state_check(ah.ImageCheck(12, 21, 5, 6, screenshot('Room(4,1)Key(False)')), '(4,1)Key', False)

classifier_3_2 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(34, 40, 16, 17, screenshot('Room(3,2)')))\
    .add_check(ah.ImageCheck(0, 17, 3, 12, screenshot('Room(3,2)')))
room_3_2 = ah.ClassifierNode((3,2))\
    .set_classifier(classifier_3_2)

classifier_4_2 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(34, 31, 7, 14, screenshot('Room(4,2)')))
room_4_2 = ah.ClassifierNode((4,2))\
    .set_classifier(classifier_4_2)

classifier_5_2 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(14, 19, 8, 19, screenshot('Room(5,2)Door(True)Door(True)Torch(True)')))\
    .add_check(ah.ImageCheck(62, 20, 6, 23, screenshot('Room(5,2)Door(True)Door(True)Torch(True)')))
room_5_2 = ah.ClassifierNode((5,2))\
    .set_classifier(classifier_5_2)\
    .add_state_check(ah.ImageCheck(52, 54, 3, 15, screenshot('Room(5,2)Door(True)Door(False)Torch(True)')), '(5,2)RightDoor', False)\
    .add_state_check(ah.ImageCheck(29, 53, 3, 16, screenshot('Room(5,2)Door(False)Door(True)Torch(True)')), '(5,2)LeftDoor', False)\
    .add_state_check(ah.ImageCheck(40, 23, 4, 5, screenshot('Room(5,2)Door(True)Door(False)Torch(False)')), '(5,2)Torch', False)

classifier_6_2 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(35, 33, 15, 13, screenshot('Room(6,2)Sword(True)')))
room_6_2 = ah.ClassifierNode((6,2))\
    .set_classifier(classifier_6_2)

classifier_7_2 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(35, 34, 15, 14, screenshot('Room(7,2)Key(True)')))\
    .add_check(ah.ImageCheck(80, 17, 4, 12, screenshot('Room(7,2)Key(True)')))
room_7_2 = ah.ClassifierNode((7,2))\
    .set_classifier(classifier_7_2)\
    .add_state_check(ah.ImageCheck(67, 21, 5, 7, screenshot('Room(7,2)Key(False)')), '(7,2)Key', False)

classifier_2_3 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(23, 34, 40, 8, screenshot('Room(2,3)Key(True)')))\
    .add_check(ah.ImageCheck(0, 20, 5, 15, screenshot('Room(2,3)Key(True)')))
room_2_3 = ah.ClassifierNode((2,3))\
    .set_classifier(classifier_2_3)\
    .add_state_check(ah.ImageCheck(40, 21, 5, 7, screenshot('Room(2,3)Key(False)')), '(2,3)Key', False)

classifier_3_3 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(37, 33, 11, 10, screenshot('Room(3,3)')))\
    .add_check(ah.ImageCheck(80, 27, 4, 17, screenshot('Room(3,3)')))
room_3_3 = ah.ClassifierNode((3,3))\
    .set_classifier(classifier_3_3)

classifier_4_3 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(67, 39, 11, 8, screenshot('Room(4,3)')))\
    .add_check(ah.ImageCheck(0, 30, 7, 14, screenshot('Room(4,3)')))
room_4_3 = ah.ClassifierNode((4,3))\
    .set_classifier(classifier_4_3)

classifier_5_3 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(34, 30, 16, 19, screenshot('Room(5,3)')))
room_5_3 = ah.ClassifierNode((5,3))\
    .set_classifier(classifier_5_3)\
    .set_agent_check(screenshot('Room(5,3)AgentPos'), 0.1, '(6,3)')

classifier_6_3 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(28, 41, 29, 9, screenshot('Room(6,3)')))
room_6_3 = ah.ClassifierNode((6,3))\
    .set_classifier(classifier_6_3)

classifier_7_3 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(35, 34, 15, 14, screenshot('Room(7,3)')))
room_7_3 = ah.ClassifierNode((7,3))\
    .set_classifier(classifier_7_3)\
    .set_agent_check(screenshot('Room(7,3)AgentPos'), 0.1, '(6,3)')

classifier_8_3 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(60, 33, 18, 17, screenshot('Room(8,3)Key(True)')))\
    .add_check(ah.ImageCheck(34, 35, 16, 6, screenshot('Room(8,3)Key(True)')))
room_8_3 = ah.ClassifierNode((8,3))\
    .set_classifier(classifier_8_3)\
    .add_state_check(ah.ImageCheck(40, 43, 4, 8, screenshot('Room(8,3)Key(False)')), '(8,3)Key', False)

classifier_1_4 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(36, 19, 12, 59, screenshot('Room(1,4)')))
room_1_4 = ah.ClassifierNode((1,4))\
    .set_classifier(classifier_1_4)

classifier_2_4 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(37, 19, 11, 24, screenshot('Room(2,4)')))
room_2_4 = ah.ClassifierNode((2,4))\
    .set_classifier(classifier_2_4)

classifier_3_4 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(27, 42, 33, 19, screenshot('Room(3,4)Door(True)Door(True)')))
room_3_4 = ah.ClassifierNode((3,4))\
    .set_classifier(classifier_3_4)\
    .add_state_check(ah.ImageCheck(71, 20, 3, 18, screenshot('Room(3,4)Door(True)Door(False)')), '(3,4)RightDoor', False)\
    .add_state_check(ah.ImageCheck(10, 20, 3, 20, screenshot('Room(3,4)Door(False)Door(False)')), '(3,4)LeftDoor', False)

classifier_4_4 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(65, 39, 9, 7, screenshot('Room(4,4)')))\
    .add_check(ah.ImageCheck(11, 39, 8, 7, screenshot('Room(4,4)')))
room_4_4 = ah.ClassifierNode((4,4))\
    .set_classifier(classifier_4_4)

classifier_5_4 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(35, 31, 16, 16, screenshot('Room(5,4)Hammer(False)')))
room_5_4 = ah.ClassifierNode((5,4))\
    .set_classifier(classifier_5_4)

classifier_6_4 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(67, 38, 10, 8, screenshot('Room(6,4)')))\
    .add_check(ah.ImageCheck(81, 20, 3, 18, screenshot('Room(6,4)')))
room_6_4 = ah.ClassifierNode((6,4))\
    .set_classifier(classifier_6_4)

classifier_7_4 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(37, 33, 11, 12, screenshot('Room(7,4)')))\
    .add_check(ah.ImageCheck(0, 20, 3, 20, screenshot('Room(7,4)')))
room_7_4 = ah.ClassifierNode((7,4))\
    .set_classifier(classifier_7_4)

classifier_8_4 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(66, 39, 9, 7, screenshot('Room(8,4)')))\
    .add_check(ah.ImageCheck(36, 30, 14, 5, screenshot('Room(8,4)')))
room_8_4 = ah.ClassifierNode((8,4))\
    .set_classifier(classifier_8_4)

classifier_9_4 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(79, 21, 5, 21, screenshot('Room(9,4)')))
room_9_4 = ah.ClassifierNode((9,4))\
    .set_classifier(classifier_9_4)

classifier_dark_room1 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(0, 19, 84, 1, screenshot('dark_room')))
room_dark_room1 = ah.ClassifierNode('dark1')\
    .set_classifier(classifier_dark_room1)

classifier_dark_room2 = ah.ImageBasedClassifier()\
    .add_check(ah.ImageCheck(0, 19, 84, 1, screenshot('dark_room')))
room_dark_room2 = ah.ClassifierNode('dark2')\
    .set_classifier(classifier_dark_room2)


abstraction_tree = ah.ClassifierTree(room_5_1, {'(5,1)Key': True,
                                                '(5,1)LeftDoor': True,
                                                '(5,1)RightDoor': True,
                                                '(5,2)RightDoor': True,
                                                '(5,2)LeftDoor': True,
                                                '(5,2)Torch': True,
                                                '(7,2)Key': True,
                                                '(2,3)Key': True,
                                                '(8,3)Key': True,
                                                '(3,4)RightDoor': True,
                                                '(3,4)LeftDoor': True
                                                })

room_4_1.connect(room_4_2)
room_4_1.connect(room_5_1)

room_5_1.connect(room_6_1)
room_5_1.connect(room_4_1)

room_6_1.connect(room_5_1)
room_6_1.connect(room_6_2)

room_3_2.connect(room_4_2)
room_3_2.connect(room_3_3)

room_4_2.connect(room_3_2)
room_4_2.connect(room_5_2)
room_4_2.connect(room_4_3)
room_4_2.connect(room_4_1)

room_5_2.connect(room_4_2)
room_5_2.connect(room_6_2)
room_5_2.connect(room_5_3)

room_6_2.connect(room_5_2)
room_6_2.connect(room_7_2)

room_7_2.connect(room_6_2)
room_7_2.connect(room_7_3)

room_2_3.connect(room_3_3)

room_3_3.connect(room_3_2)
room_3_3.connect(room_2_3)

room_4_3.connect(room_4_2)
room_4_3.connect(room_5_3)

room_5_3.connect(room_4_3)
room_5_3.connect(room_6_3)
room_5_3.connect(room_5_4)
room_5_3.connect(room_5_2)

room_6_3.connect(room_5_3)
room_6_3.connect(room_7_3)

room_7_3.connect(room_8_3)
room_7_3.connect(room_6_3)
room_7_3.connect(room_7_4)
room_7_3.connect(room_7_2)

room_8_3.connect(room_7_3)
room_8_3.connect(room_8_4)

room_1_4.connect(room_2_4)

room_2_4.connect(room_1_4)
room_2_4.connect(room_3_4)

room_3_4.connect(room_2_4)
room_3_4.connect(room_2_4)

room_4_4.connect(room_3_4)
room_4_4.connect(room_5_4)

room_5_4.connect(room_4_4)
room_5_4.connect(room_6_4)
room_5_4.connect(room_5_3)

room_6_4.connect(room_5_4)

room_7_4.connect(room_7_3)
room_7_4.connect(room_8_4)

room_8_4.connect(room_7_4)
room_8_4.connect(room_8_3)
room_8_4.connect(room_9_4)

room_9_4.connect(room_8_4)

room_dark_room1.connect(room_5_3)
room_dark_room2.connect(room_8_3)
room_dark_room2.connect(room_7_3)



room_list = [(1,4),(2,3),(2,4),(3,2), (3,3), (3,4), (4,1), (4,2), (4,3), (4,4), (5,1),
             (5,2), (5,3), (5,4), (6,1), (6,2),(6,3), (6,4), (7,2), (7,3), (7,4), (8,3),
             (8,4), (9,4)]
room_mapping = dict([(room, i) for i, room in enumerate(room_list)])

sectors = [(x, y) for x in range(3) for y in range(3)]
sector_mapping = dict([(sector, i) for i, sector in enumerate(sectors)])

def room_to_onehot(room):
    onehot = np.zeros(shape=len(room_list), dtype=np.uint8)
    onehot[room_mapping[room]] = 1
    return onehot

def sector_to_onehot(sector):
    onehot = np.zeros(shape=len(sectors), dtype=np.uint8)
    onehot[sector_mapping[sector]] = 1
    return onehot

def montezuma_abstraction_vector(abstract_state):
    room = room_to_onehot(abstract_state.name)
    sector = sector_to_onehot(abstract_state.sector)
    rest = []
    for (key, value) in abstract_state.global_state_list:
        rest.append(2*int(value)-1)
    return np.concatenate([room, sector, rest], axis=0)

