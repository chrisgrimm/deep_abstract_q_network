import cv2
import numpy as np
import os

base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../screenshots')
screenshot = lambda name: cv2.imread(os.path.join(base_path, name+'.png'))


class ImageCheck(object):

    def __init__(self, pos_x, pos_y, size_x, size_y, image):
        self.ltx = pos_x
        self.lty = pos_y
        self.rbx = pos_x+size_x
        self.rby = pos_y+size_y
        self.image = image[pos_y:pos_y+size_y, pos_x:pos_x+size_x, 0]
        assert self.ltx < self.rbx
        assert self.lty < self.rby
        assert self.image.shape[0] == size_y and self.image.shape[1] == size_x

    def passes_check(self, screen):
        return np.array_equal(screen[self.lty:self.rby, self.ltx:self.rbx], self.image)


class ImageBasedClassifier(object):

    def __init__(self):
        self.checks = []
        self.tree_checks = []

    def add_check(self, check):
        self.checks.append(check)
        return self

    def passes_check(self, screen):
        screen_check_result = all(check.passes_check(screen) for check in self.checks)
        tree_check_result = all(tree.global_state[key] == value for tree, key, value in self.tree_checks)
        return screen_check_result and tree_check_result




class ClassifierTree(object):

    def __init__(self, starting_node, starting_state):
        self.starting_node = starting_node
        self.starting_state = starting_state
        self.env = None
        self.reset()

    def setEnv(self, env):
        self.env = env

    def reset(self):
        self.global_state = self.starting_state.copy()
        self.current_node = self.starting_node
        self.last_node = self.starting_node
        self.agent_sector = (1, 2)
        self.death_counter_triggered = False

    def should_perform_sector_check(self):
        ram = self.env.getRAM()
        is_falling = ram[88] > 0
        death_counter_active = ram[55] > 0
        death_sprite_active = ram[54] == 6
        is_walking_or_on_stairs = ram[53] in [0, 10, 8, 18]
        # also need in_jump because there is a frame where the agent is walking when it jumps on the rope
        in_jump = ram[56] == 255
        should_check = (is_walking_or_on_stairs and not in_jump and not is_falling and not death_counter_active and not death_sprite_active)
        return should_check

    def bout_to_get_murked(self):
        ram = self.env.getRAM()
        is_falling = ram[88] > 0
        death_counter_active = ram[55] > 0
        death_sprite_active = ram[54] == 6
        return is_falling or death_counter_active or death_sprite_active

    def get_agent_sector(self):
        if self.should_perform_sector_check():
            ram = self.env.getRAM()
            pos_x, pos_y = (ram[0xAA - 0x80] - 0x01) / float(0x98 - 0x01), (ram[0xAB - 0x80] - 0x86) / float(0xFF - 0x86)
            self.agent_sector = np.clip(int(3 * pos_x), 0, 2), np.clip(int(3 * pos_y), 0, 2)
        return self.agent_sector

    def perform_state_checks(self, screen):
        for classifier, (key, value) in self.current_node.state_checks:
            if classifier.passes_check(screen):
                self.global_state[key] = value

    def perform_transition_checks(self, screen, ):
        for node in self.current_node.neighbors:
            if node.passes_check(screen, self.current_node.name):
                self.last_node = self.current_node
                self.current_node = node
                break

    def update_state(self, screen):
        self.perform_state_checks(screen)
        self.perform_transition_checks(screen)

    def get_abstract_state(self):
        abs_state = AbstractState(self.current_node.name, self.get_agent_sector(), self.global_state.copy())
        return abs_state


class AbstractState(object):

    def __init__(self, name, sector, global_state):
        self.name = name
        self.sector = sector
        self.global_state_list = sorted(global_state.items())

    def __str__(self):
        room = self.name
        sector = self.sector
        bools = ''.join(str(int(y)) for x, y in self.global_state_list)
        return str((room, sector)) + bools

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return str(self) == str(other)

    def __ne__(self, other):
        if not isinstance(other, self.__class__):
            return True
        return str(self) != str(other)

    def __hash__(self):
        return hash(str(self))

class ClassifierNode(object):

    def __init__(self, name):
        self.cls = None
        self.state_checks = []
        self.agent_check = None
        self.neighbors = set()
        self.name = name

    def set_classifier(self, classifier):
        self.cls = classifier
        return self

    def add_state_check(self, check, key, value):
        self.state_checks.append((check, (key, value)))
        return self

    def set_agent_check(self, desired_screen, delta, required_prev_state):
        self.agent_check = (desired_screen, delta, required_prev_state)
        return self

    def connect(self, other_node):
        self.neighbors.add(other_node)
        other_node.neighbors.add(self)
        return self

    def passes_check(self, screen, previous_state):
        image_checks = self.cls.passes_check(screen)
        if self.agent_check is not None and self.agent_check[2] == previous_state:
            (desired_screen, delta, required_prev_state) = self.agent_check
            agent_conv = screenshot('Room(7,3)AgentPos')[29:29 + 9, 0:0 + 5, 0]
            detector_image_1 = cv2.matchTemplate(screen, agent_conv, cv2.TM_SQDIFF_NORMED)
            [pos_y_screen, pos_x_screen] = np.array(np.unravel_index(np.argmin(detector_image_1), detector_image_1.shape)) / 84.
            detector_image_2 = cv2.matchTemplate(desired_screen[:,:,0], agent_conv, cv2.TM_SQDIFF_NORMED)
            [pos_y_desired, pos_x_desired] = np.array(np.unravel_index(np.argmin(detector_image_2), detector_image_2.shape)) / 84.
            passes_agent_check = (np.sqrt((pos_x_desired - pos_x_screen)**2 + (pos_y_desired - pos_y_screen)**2) < delta)
            # print pos_x_screen, pos_y_screen
            return passes_agent_check and image_checks
        return image_checks