import random
import cv2
import time
import numpy as np
import copy
import os
import json
import numpyarrayencoder
from PIL import Image
from time import sleep

np.set_printoptions(threshold=np.inf)

# Tetris game class
class Tetris:

    '''Tetris game class'''

    # BOARD
    MAP_EMPTY = 0
    MAP_BLOCK = 1
    MAP_PLAYER = 2
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20

    SAMPLE = 0
    JSON_OUTPUT = False
    IMAGE_OUTPUT = True

    TETROMINOS = {
        0: { # I
            0: [(0,0), (1,0), (2,0), (3,0)],
            90: [(1,0), (1,1), (1,2), (1,3)],
            180: [(3,0), (2,0), (1,0), (0,0)],
            270: [(1,3), (1,2), (1,1), (1,0)],
        },
        1: { # T
            0: [(1,0), (0,1), (1,1), (2,1)],
            90: [(0,1), (1,2), (1,1), (1,0)],
            180: [(1,2), (2,1), (1,1), (0,1)],
            270: [(2,1), (1,0), (1,1), (1,2)],
        },
        2: { # L
            0: [(1,0), (1,1), (1,2), (2,2)],
            90: [(0,1), (1,1), (2,1), (2,0)],
            180: [(1,2), (1,1), (1,0), (0,0)],
            270: [(2,1), (1,1), (0,1), (0,2)],
        },
        3: { # J
            0: [(1,0), (1,1), (1,2), (0,2)],
            90: [(0,1), (1,1), (2,1), (2,2)],
            180: [(1,2), (1,1), (1,0), (2,0)],
            270: [(2,1), (1,1), (0,1), (0,0)],
        },
        4: { # Z
            0: [(0,0), (1,0), (1,1), (2,1)],
            90: [(0,2), (0,1), (1,1), (1,0)],
            180: [(2,1), (1,1), (1,0), (0,0)],
            270: [(1,0), (1,1), (0,1), (0,2)],
        },
        5: { # S
            0: [(2,0), (1,0), (1,1), (0,1)],
            90: [(0,0), (0,1), (1,1), (1,2)],
            180: [(0,1), (1,1), (1,0), (2,0)],
            270: [(1,2), (1,1), (0,1), (0,0)],
        },
        6: { # O
            0: [(1,0), (2,0), (1,1), (2,1)],
            90: [(1,0), (2,0), (1,1), (2,1)],
            180: [(1,0), (2,0), (1,1), (2,1)],
            270: [(1,0), (2,0), (1,1), (2,1)],
        }
    }

    COLORS = {
        0: (255, 255, 255),
        1: (247, 64, 99),
        2: (0, 167, 247),
    }


    def __init__(self, target_image=None):
        self.reset()
        # set to ensure only the first time a sample is saved a new folder is created
        self.create_folder = True

        # set to ensure a unique generated folder name
        self.folder_n = 1

        # set a target if present
        if target_image:
            self._set_target_board(target_image)

    
    def reset(self):
        '''Resets the game, returning the current state'''
        self.board = [[0] * Tetris.BOARD_WIDTH for _ in range(Tetris.BOARD_HEIGHT)]
        self.game_over = False
        self.bag = list(range(len(Tetris.TETROMINOS)))
        random.shuffle(self.bag)
        self.next_piece = self.bag.pop()
        self._new_round()
        self.score = 0
        return self._get_board_props(self.board)

    def _set_target_board(self, image):
        self.target = self._convert_image_to_array(image)

    def _convert_image_to_array(self, image):
        # Load the image
        img = Image.open(image)

        # Convert to grayscale
        img = img.convert('L')

        # Convert to numpy array
        img_array = np.array(img)

        # Define a threshold to distinguish between black and white
        threshold = 128

        # Create a binary array: 0 for white (>= threshold), 1 for black (< threshold)
        binary_array = np.where(img_array < threshold, 1, 0)

        # print(binary_array)

        binary_list = [list(i) for i in list(binary_array)]

        return binary_list
    
    def _convert_array_to_image(self, current_board, n, num_samples, recursion):
        postfix = ""
        if num_samples:
            postfix = "_r"
        output_dir = f"samples{postfix}_{self.folder_n}"
        # only create folder initially, subsequent calls should add to that folder
        if self.create_folder:
            self.create_folder = False

            folder_created = False
            while not folder_created:
                try:
                    os.makedirs(output_dir)
                except:
                    self.folder_n += 1
                    output_dir = f"samples{postfix}_{self.folder_n}"
                else:
                    folder_created = True

        # print(current_board)

        array = np.array(current_board, dtype=np.uint8)
        image_array = 255 - array * 255
        image = Image.fromarray(image_array)

        file_path = os.path.join(output_dir, f"sample_{n}_score_{recursion}.png")
        image.save(file_path)
        print("saved to ", file_path)
        # image.show()

    def _convert_single_sample_to_output(self, current_board, num_samples, recursion, json, image):
        if image:
            self._convert_array_to_image(current_board, self.SAMPLE, num_samples, recursion)
            self.SAMPLE += 1
        if json:
            self.export_samples_to_json(recursion, [current_board])


    def _get_rotated_piece(self):
        '''Returns the current piece, including rotation'''
        # print("aaaa")
        # print(self.current_rotation)
        # print(Tetris.TETROMINOS)
        return Tetris.TETROMINOS[self.current_piece][self.current_rotation]


    def _get_complete_board(self):
        '''Returns the complete board, including the current piece'''
        piece = self._get_rotated_piece()
        piece = [np.add(x, self.current_pos) for x in piece]
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y][x] = Tetris.MAP_PLAYER
        return board


    def get_game_score(self):
        '''Returns the current game score.

        Each block placed counts as one.
        For lines cleared, it is used BOARD_WIDTH * lines_cleared ^ 2.
        '''
        return self.score
    

    def _new_round(self):
        '''Starts a new round (new piece)'''
        # Generate new bag with the pieces
        if len(self.bag) == 0:
            self.bag = list(range(len(Tetris.TETROMINOS)))
            random.shuffle(self.bag)
        
        self.current_piece = self.next_piece
        self.next_piece = self.bag.pop()
        self.current_pos = [3, 0]
        self.current_rotation = 0

        if self._check_collision(self._get_rotated_piece(), self.current_pos):
            self.game_over = True


    def _check_collision(self, piece, pos):
        '''Check if there is a collision between the current piece and the board'''
        for x, y in piece:
            x += pos[0]
            y += pos[1]
            if x < 0 or x >= Tetris.BOARD_WIDTH \
                    or y < 0 or y >= Tetris.BOARD_HEIGHT \
                    or self.board[y][x] == Tetris.MAP_BLOCK:
                return True
        return False


    def _rotate(self, angle):
        '''Change the current rotation'''
        r = self.current_rotation + angle

        if r == 360:
            r = 0
        if r < 0:
            r += 360
        elif r > 360:
            r -= 360

        self.current_rotation = r


    def _add_piece_to_board(self, piece, pos):
        '''Place a piece in the board, returning the resulting board'''  
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y + pos[1]][x + pos[0]] = Tetris.MAP_BLOCK
        return board


    def _clear_lines(self, board):
        '''Clears completed lines in a board'''
        # Check if lines can be cleared
        lines_to_clear = [index for index, row in enumerate(board) if sum(row) == Tetris.BOARD_WIDTH]
        if lines_to_clear:
            board = [row for index, row in enumerate(board) if index not in lines_to_clear]
            # Add new lines at the top
            for _ in lines_to_clear:
                board.insert(0, [0 for _ in range(Tetris.BOARD_WIDTH)])
        return len(lines_to_clear), board


    def _number_of_holes(self, board):
        '''Number of holes in the board (empty sqquare with at least one block above it)'''
        holes = 0

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] != Tetris.MAP_BLOCK:
                i += 1
            holes += len([x for x in col[i+1:] if x == Tetris.MAP_EMPTY])

        return holes


    def _bumpiness(self, board):
        '''Sum of the differences of heights between pair of columns'''
        total_bumpiness = 0
        max_bumpiness = 0
        min_ys = []

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] != Tetris.MAP_BLOCK:
                i += 1
            min_ys.append(i)
        
        for i in range(len(min_ys) - 1):
            bumpiness = abs(min_ys[i] - min_ys[i+1])
            max_bumpiness = max(bumpiness, max_bumpiness)
            total_bumpiness += abs(min_ys[i] - min_ys[i+1])

        return total_bumpiness, max_bumpiness


    def _height(self, board):
        '''Sum and maximum height of the board'''
        sum_height = 0
        max_height = 0
        min_height = Tetris.BOARD_HEIGHT

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] == Tetris.MAP_EMPTY:
                i += 1
            height = Tetris.BOARD_HEIGHT - i
            sum_height += height
            if height > max_height:
                max_height = height
            elif height < min_height:
                min_height = height

        return sum_height, max_height, min_height


    def _get_board_props(self, board):
        '''Get properties of the board'''
        lines, board = self._clear_lines(board)
        holes = self._number_of_holes(board)
        total_bumpiness, max_bumpiness = self._bumpiness(board)
        sum_height, max_height, min_height = self._height(board)
        return [lines, holes, total_bumpiness, sum_height]

    def generate_samples(self, depth, num_samples=0):
        # depth: level of recursion, sample_count: number of random samples. If None, then generate all possible samples from recursion 0 up to recursion depth
        # set level of recursion (explodes exponentially)
        self.depth = depth
        self.samples = []
        
        
        if not num_samples:

            print(f"Generating all possible samples up to depth {depth}")
            
            current_board = self.target.copy()

            # generate all possible states
            print("recursion: ", 0)
            self.get_past_states(current_board, 0)

        else:

            print(f"Generating a total of {num_samples} samples with max depth {depth}")

            n = 0
            while n < num_samples:
                
                current_board = self.target.copy()
                # set a random depth to play towards
                self.depth = random.randint(1, depth)

                print(f"sample {n} depth {self.depth}")
                self.get_past_states_random(current_board, n, 0)
                n += 1
            
        # for n, sample in enumerate(self.samples):
        #     # save to image file
        #     self._convert_array_to_image(sample, n, num_samples)

        self.export_samples_to_json(depth, num_samples)

    def get_past_states_random(self, current_board, num_samples, recursion):
        # - add no line / line at every possible row (up to the highest placed tiles): shift everything else upward
        # - check for each block all possible layouts
        # - if exists and placeable:
        #   - remove block and add state + distance (= recursion) to samples
        #   - recurse

        # board coordinates start at top left, (y, x), increasing y goes down, increasing x goes right


        # continue generating samples if either all possible samples must be generated, or the random playout is not finished
        if recursion >= self.depth:
            # self.samples.append((current_board, np.int32(recursion)))
            self._convert_single_sample_to_output(current_board, num_samples, recursion, self.JSON_OUTPUT, self.IMAGE_OUTPUT)

        else:
            boards = []
            #set the height to where lines can be added
            #max height of n -> from n-1 up until 19 (bottom of board) lines can be added and rows 0 to n-1 shift up
            max_height = 21
            #if board full, then no lines can be added
            if not any(current_board[0]):
                for i, row in enumerate(current_board):
                    if any(row):
                        max_height = i - 1
                        break
                    #if board empty, bottom line can be added
                    max_height = 20
                # random playout
                # check if depth is reached
            

            for i in range(max_height, 21):
                new_board = copy.deepcopy(current_board)
                self.add_line(new_board, i)
                boards.append(new_board)

            options = []
            options_line_added = []

            # print(current_board)
            #seperate regular options from options where a line is added
            # print("current")
            current_samples = self.get_removing_pieces_samples(current_board)
            # print(current_samples)
            for sample in current_samples:
                options.append(sample)


            for board in boards:
                current_line_samples = self.get_removing_pieces_samples(board)
                for sample in current_line_samples:
                    options_line_added.append(sample)
                

            # if not random.randint(0, 9):
            #     random_row = random.randint(max_height, 20)
            #     # no copy needed, because only one playout is done
            #     self.add_line(current_board, random_row)


            # select random playout
            # 10% chance to add a line after checking if either lists are empty
            choice = random.randint(0,4)
            # print(choice)
            # print(not options)
            # print(options)
            # time.sleep(1)


            # r_14 = 20% chance
            # r_15 = 40% chance


            if (choice <= 1 and len(options_line_added) != 0) or not options:
                current_board = random.choice(options_line_added)
            # 90% chance to not add a line
            else:
                current_board = random.choice(options)
            # print(np.array(current_board))
            # self._convert_array_to_image([current_board, 0], recursion, 1)
            del boards
            del options
            print("recursion: ", recursion)
            # continue until recursion has reached depth
            self.get_past_states_random(current_board, num_samples+1, recursion+1)

    #export boards to json. Takes samples as input array. If None, then use self.samples
    def export_samples_to_json(self, recursion, num_samples, samples=None):
        # convert to JSON for storage
        if not samples:
            samples = self.samples
        sample_dict = [{"array": arr, "score": int(score)} for arr, score in samples]
        print(sample_dict)
        json_data = json.dumps(sample_dict, indent=4, cls=numpyarrayencoder.NumpyArrayEncoder)

        postfix = f"{num_samples}"
        if num_samples == 0:
            postfix = ""

        with open (f'generated_samples_recursion_{recursion}_{postfix}.json', 'w') as file:
            file.write(json_data)

    def add_line(self, current_board, index):
        #make sure the top row is empty before adding a line
        if not any(current_board[0]):
            current_board.insert(index, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            current_board.pop(0)

    def get_past_states(self, current_board, recursion):
        # - add no line / line at every possible row (up to the highest placed tiles): shift everything else upward
        # - check for each block all possible layouts
        # - if exists and placeable:
        #   - remove block and add state + distance (= recursion) to samples
        #   - recurse

        # board coordinates start at top left, (y, x), increasing y goes down, increasing x goes right

        #set the height to where lines can be added
        #max height of n -> from n-1 up until 19 (bottom of board) lines can be added and rows 0 to n-1 shift up
        max_height = 21
        boards = [current_board]
        #if board full, then no lines can be added
        if not any(current_board[0]):
            for i, row in enumerate(current_board):
                if any(row):
                    max_height = i - 1
                    break
                #if board empty, bottom line can be added
                max_height = 20

        # for loop to add all potential boards
        # range 22: 20 = add line below bottom row of board, 21 = don't add any line
        for i in range(max_height, 22):
            if i != 21:
                new_board = copy.deepcopy(current_board)
                self.add_line(new_board, i)
            else:
                new_board = current_board
            print("r ", recursion, " board ", i)
            current_samples = self.get_removing_pieces_samples(new_board)
            for sample in current_samples:
                # self.samples.append((sample, np.int32(recursion)))
                self._convert_single_sample_to_output(sample, 0, recursion, self.JSON_OUTPUT, self.IMAGE_OUTPUT)
                if recursion < self.depth:
                    print("recursion: ", recursion)
                    self.get_past_states(sample, recursion+1)
            # return self.samples

        # # for loop to add all potential boards where a line was cleared
        # # range 21: 20 = add line below bottom row of board
        # for i in range(max_height, 21):
        #     new_board = copy.deepcopy(current_board)
        #     self.add_line(new_board, i)
        #     boards.append(new_board)

        # for i, board in enumerate(boards):
        #     print("r ", recursion, " board ", i)
        #     current_samples = self.get_removing_pieces_samples(board)
        #     for sample in current_samples:
        #         self.samples.append((sample, np.int32(recursion)))
        #     print("samples: ", len(current_samples))

        #     if recursion < self.depth:
        #         print("recursion: ", recursion)
        #         for board in current_samples:
        #             self.get_past_states(board, recursion+1)
        #     # return self.samples


    def get_removing_pieces_samples(self, current_board):
        current_samples = []
        # print(self.TETROMINOS)
        # print(np.array(current_board))
        full_line = -1
        for i, line in enumerate(current_board):
            if all(tile == 1 for tile in line):
                full_line = i
                break
        for tetromino in self.TETROMINOS:
            # print(tetromino)
            #coordinates of tetrominos
            # [(0, 0), (1, 0), (2, 0), (3, 0)]
            # [(1, 0), (0, 1), (1, 1), (2, 1)]
            # [(1, 0), (1, 1), (1, 2), (2, 2)]
            # [(1, 0), (1, 1), (1, 2), (0, 2)]
            # [(0, 0), (1, 0), (1, 1), (2, 1)]
            # [(2, 0), (1, 0), (1, 1), (0, 1)]
            # [(1, 0), (2, 0), (1, 1), (2, 1)]
            # print(self.TETROMINOS[tetromino][0])

            # manually set the offset per rotated piece to ensure that the piece has one tile at (0, 0)
            # returns (y1, x1),...,(yn, xn), where n = number of rotations
            def get_offset(piece):
                if piece == 0: #I
                    return [(0, 0),(-1, 0)]
                elif piece == 1: #T
                    return [(0, -1),(-1, 0),(0, -1),(-1, 0)]
                elif piece == 2: #J
                    return [(-1, 0),(0, -1),(0, 0),(0, -1)]
                elif piece == 3: #L
                    return [(-1, 0),(0, -1),(-1, 0),(0, 0)]
                elif piece == 4: #S
                    return [(0, 0),(-1, 0),(0, 0),(-1, 0)]
                elif piece == 5: #Z
                    return [(-1, 0),(0, 0),(-1, 0),(0, 0)]
                elif piece == 6: #O
                    return [(0, 0)]

            if tetromino == 6: 
                rotations = [0]
            elif tetromino == 0:
                rotations = [0, 90]
            else:
                rotations = [0, 90, 180, 270]
            for r, rotation in enumerate(rotations):
                # print("tetromino ", tetromino)
                # print("rotation ", rotation)
                s = 0
                rotated_piece = self.TETROMINOS[tetromino][rotation]
                # print("before offset: ", rotated_piece)
                offsets = get_offset(tetromino)
                # print(offsets[r])
                # print(rotated_piece)
                rotated_piece = [(x + offsets[r][0], y + offsets[r][1]) for x, y in rotated_piece]
                # print("after offset: ", rotated_piece)
                

                # self.p = False
                # if tetromino == 0 and rotation == 90:
                #     self.p = True

                # print("rotated: ", rotated_piece)
                # fit all pieces in the playing field
                for i, row in enumerate(current_board):
                    # check if there exists tiles in row
                    if any(row):
                        # go through all columns
                        # print(row)
                        for j, col in enumerate(row):
                            # only check if a tile is present
                            if col:
                                # print(col)
                                #try to fit pieces in blocks
                                if self.piece_fits_in_board(rotated_piece, current_board, i, j, full_line):
                                    s += 1
                                    copy_board = copy.deepcopy(current_board)
                                    self.remove_piece(rotated_piece, copy_board, i, j)
                                    current_samples.append(copy_board)
                                    # print("append")
                # print("samples ", s)
                # print()

        # print("samples")
        # print(len(current_samples))
        return current_samples

    def remove_piece(self, piece, copy_board, row, col):
        # print("before ", piece, " - ", row, " - ", col)
        # print(np.array(copy_board))
        for tile in piece:
            copy_board[row + tile[0]][col + tile[1]] = 0
        # print("after ", piece, " - ", row, " - ", col)
        # print(np.array(copy_board))

    def piece_fits_in_board(self, piece, current_board, row, col, full_line):
        # For every tile, attempt to fit piece into present tiles
        # if self.p:
            # print(piece)
        # print("fit")
        # print(row)
        # print(col)
        # print(piece)
        tile_on_line = False
        for tile in piece:
            # if self.p:
                # print(row + tile[0], col + tile[1])
            # print(len(current_board))
            # print(len(current_board[0]))
            # print(row+tile[0])
            # print(col+tile[1])
            if row + tile[0] > len(current_board)-1 or col + tile[1] > len(current_board[0])-1:
                # if self.p:
                #     print(len(current_board))
                #     print(len(current_board[0]))
                # print("outside")
                # time.sleep(1)
                return False
            # coordinates are relative to a present tile on the board
            # print(row+tile[0])
            # print(len(current_board))
            # print(col+tile[1])
            # print(len(current_board[0]))
            if not current_board[row + tile[0]][col + tile[1]]:
                # print("false")
                # time.sleep(1)
                return False
            
            #If a line was added, ensure that the piece that is removed was placed on the line
            if row + tile[0] == full_line:
                tile_on_line = True
        if full_line == -1 or tile_on_line:
            return True
        return False
    
    def has_next_states(self):
        for piece_id in range(7):
            if piece_id == 6: 
                rotations = [0]
            elif piece_id == 0:
                rotations = [0, 90]
            else:
                rotations = [0, 90, 180, 270]

            # For all rotations
            for rotation in rotations:
                piece = Tetris.TETROMINOS[piece_id][rotation]
                min_x = min([p[0] for p in piece])
                max_x = max([p[0] for p in piece])

                # For all positions
                for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                    pos = [x, 0]

                    # Drop piece
                    while not self._check_collision(piece, pos):
                        pos[1] += 1
                    pos[1] -= 1

                    # Valid move
                    if pos[1] >= 0:
                        return True
        return False


    def get_next_states(self, pieces = None):
        '''Get all possible next states'''
        states = {}
        if pieces is not None:
            states = []
        piece_id = self.current_piece
        
        if piece_id == 6: 
            rotations = [0]
        elif piece_id == 0:
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]

        # For all rotations
        for rotation in rotations:
            piece = Tetris.TETROMINOS[piece_id][rotation]
            min_x = min([p[0] for p in piece])
            max_x = max([p[0] for p in piece])

            # For all positions
            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                pos = [x, 0]

                # Drop piece
                while not self._check_collision(piece, pos):
                    pos[1] += 1
                pos[1] -= 1

                # Valid move
                if pos[1] >= 0:
                    if pieces is None:
                        states[(x, rotation)] = self._get_board_props(self.board)
                    else:
                        states.append((piece, pos))
                        # print(f"appended {piece}, {pos}")

        return states


    def get_state_size(self):
        '''Size of the state'''
        return 4

    # TODO: calculate new reward function
    def get_reachability_score(self):
        return self.attempt1()
    
    # raw state space difference per tile
    def attempt1(self):
        board = self._get_complete_board()
        score = 0
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == self.target[i][j]:
                    score += 1
                else:
                    score -= 1
        return score
    
    # def run_mcts(self):
    #     board = self._get_complete_board()
    #     model = import_regression_model(self.)
    #     agent = MCAgent(board, self.PLAYOUTS, )


    def play(self, x, rotation, render=False, render_delay=None, reachability=False):
        '''Makes a play given a position and a rotation, returning the reward and if the game is over'''
        self.current_pos = [x, 0]
        self.current_rotation = rotation

        # Drop piece
        while not self._check_collision(self._get_rotated_piece(), self.current_pos):
            if render:
                self.render()
                if render_delay:
                    sleep(render_delay)
            self.current_pos[1] += 1
        self.current_pos[1] -= 1

        # Update board and calculate score        
        self.board = self._add_piece_to_board(self._get_rotated_piece(), self.current_pos)
        lines_cleared, self.board = self._clear_lines(self.board)
        score = 0
        if reachability:
            score = self.get_reachability_score()
            self.score = score
        else:
            score = 1 + (lines_cleared ** 2) * Tetris.BOARD_WIDTH
            self.score += score

        # Start new round
        self._new_round()
        if self.game_over:
            score -= 2

        return score, self.game_over


    def render(self):
        '''Renders the current board'''
        img = [Tetris.COLORS[p] for row in self._get_complete_board() for p in row]
        img = np.array(img).reshape(Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH, 3).astype(np.uint8)
        img = img[..., ::-1] # Convert RGB to BGR (used by cv2)
        img = Image.fromarray(img, 'RGB')
        img = img.resize((Tetris.BOARD_WIDTH * 25, Tetris.BOARD_HEIGHT * 25), Image.NEAREST)
        img = np.array(img)
        cv2.putText(img, str(self.score), (22, 22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.imshow('image', np.array(img))
        cv2.waitKey(1)