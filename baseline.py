#####################################################
# This is the Tetris AI provided by the Nintaco API #
# This API is used as the baseline API where the    #
# goal is to survive as long as possible            #
#####################################################

import sys

from nintaco import nintaco

PLAYFIELD_WIDTH = 10
PLAYFIELD_HEIGHT = 20
TETRIMINOS_SEARCHED = 2
  
WEIGHTS = [
  1.0,
  12.885008263218383,
  15.842707182438396,
  26.89449650779595,
  27.616914062397015,
  30.18511071927904,
]

NONE = -1
T = 0
J = 1
Z = 2
O = 3
S = 4
L = 5
I = 6

PATTERNS = [
  [ [ [ -1,  0 ], [  0,  0 ], [  1,  0 ], [  0,  1 ], ],    # Td (spawn)
    [ [  0, -1 ], [ -1,  0 ], [  0,  0 ], [  0,  1 ], ],    # Tl    
    [ [ -1,  0 ], [  0,  0 ], [  1,  0 ], [  0, -1 ], ],    # Tu
    [ [  0, -1 ], [  0,  0 ], [  1,  0 ], [  0,  1 ], ], ], # Tr   

  [ [ [ -1,  0 ], [  0,  0 ], [  1,  0 ], [  1,  1 ], ],    # Jd (spawn)
    [ [  0, -1 ], [  0,  0 ], [ -1,  1 ], [  0,  1 ], ],    # Jl
    [ [ -1, -1 ], [ -1,  0 ], [  0,  0 ], [  1,  0 ], ],    # Ju
    [ [  0, -1 ], [  1, -1 ], [  0,  0 ], [  0,  1 ], ], ], # Jr   

  [ [ [ -1,  0 ], [  0,  0 ], [  0,  1 ], [  1,  1 ], ],    # Zh (spawn) 
    [ [  1, -1 ], [  0,  0 ], [  1,  0 ], [  0,  1 ], ], ], # Zv   

  [ [ [ -1,  0 ], [  0,  0 ], [ -1,  1 ], [  0,  1 ], ], ], # O  (spawn)   

  [ [ [  0,  0 ], [  1,  0 ], [ -1,  1 ], [  0,  1 ], ],    # Sh (spawn)
    [ [  0, -1 ], [  0,  0 ], [  1,  0 ], [  1,  1 ], ], ], # Sv   

  [ [ [ -1,  0 ], [  0,  0 ], [  1,  0 ], [ -1,  1 ], ],    # Ld (spawn)
    [ [ -1, -1 ], [  0, -1 ], [  0,  0 ], [  0,  1 ], ],    # Ll
    [ [  1, -1 ], [ -1,  0 ], [  0,  0 ], [  1,  0 ], ],    # Lu
    [ [  0, -1 ], [  0,  0 ], [  0,  1 ], [  1,  1 ], ], ], # Lr      

  [ [ [ -2,  0 ], [ -1,  0 ], [  0,  0 ], [  1,  0 ], ],    # Ih (spawn)    
    [ [  0, -2 ], [  0, -1 ], [  0,  0 ], [  0,  1 ], ], ], # Iv      
]

ORIENTATION_IDS = [ 0x02, 0x03, 0x00, 0x01, 0x07, 0x04, 0x05, 0x06, 0x08, 0x09, 
                    0x0A, 0x0B, 0x0C, 0x0E, 0x0F, 0x10, 0x0D, 0x12, 0x11 ]
                    
class Point(object):
  def __init__(self, x = 0, y = 0):
    self.x = x
    self.y = y
               
class Orientation(object):
  def __init__(self):
    self.squares = [Point() for x in range(4)]
    self.minX = 0
    self.minY = 0
    self.maxY = 0
    self.orientationID = 0

ORIENTATIONS = [[] for i in range(len(PATTERNS))]
idIndex = 0
for i in range(len(PATTERNS)):
  for j in range(len(PATTERNS[i])):
    tetrimino = Orientation()
    ORIENTATIONS[i].append(tetrimino)
    minX = sys.maxint
    maxX = -sys.maxint - 1
    maxY = -sys.maxint - 1 
    for k in range(4):
      p = PATTERNS[i][j][k]
      tetrimino.squares[k].x = p[0]
      tetrimino.squares[k].y = p[1]
      minX = min(minX, p[0])
      maxX = max(maxX, p[0])
      maxY = max(maxY, p[1]) 
    tetrimino.minX = -minX;
    tetrimino.maxX = PLAYFIELD_WIDTH - maxX - 1
    tetrimino.maxY = PLAYFIELD_HEIGHT - maxY - 1
    tetrimino.orientationID = ORIENTATION_IDS[idIndex]
    idIndex += 1
    
class PlayfieldEvaluation(object):
  def __init__(self):
    self.holes = 0
    self.columnTransitions = 0
    self.rowTransitions = 0
    self.wells = 0
    
class State(object):
  def __init__(self, x, y, rotation):
    self.x = x
    self.y = y
    self.rotation = rotation
    self.visited = 0
    self.predecessor = None
    
spareRows = [[NONE for i in range(PLAYFIELD_WIDTH + 1)]
    for i in range(8 * TETRIMINOS_SEARCHED)]
columnDepths = [0 for i in range(PLAYFIELD_WIDTH)]
spareIndex = 0

def createPlayfield():
  return [[NONE for i in range(PLAYFIELD_WIDTH + 1)] 
      for i in range(PLAYFIELD_HEIGHT)]
      
def lockTetrimino(playfield, tetriminoType, state):
  for square in ORIENTATIONS[tetriminoType][state.rotation].squares:
    y = state.y + square.y
    if y >= 0:
      playfield[y][state.x + square.x] = tetriminoType
      playfield[y][AI.PLAYFIELD_WIDTH] += 1
  
  startRow = state.y - 2
  endRow = state.y + 1
  
  if startRow < 1:
    startRow = 1
  if endRow >= PLAYFIELD_HEIGHT:
    endRow = PLAYFIELD_HEIGHT - 1
    
  for y in range(startRow, endRow + 1):
    if playfield[y][PLAYFIELD_WIDTH] == PLAYFIELD_WIDTH: 
      clearedRow = playfield[y]
      for i in range(y, 0, -1):
        playfield[i] = playfield[i - 1]
      for x in range(PLAYFIELD_WIDTH):
        clearedRow[x] = NONE          
      clearedRow[PLAYFIELD_WIDTH] = 0
      playfield[0] = clearedRow
      
def evaluatePlayfield(playfield):
  global e
  
  for x in range(PLAYFIELD_WIDTH):
    columnDepths[x] = PLAYFIELD_HEIGHT - 1
    for y in range(PLAYFIELD_HEIGHT):
      if playfield[y][x] != NONE:
        columnDepths[x] = y
        break

  e.wells = 0
  for x in range(PLAYFIELD_WIDTH):
    minY = 0
    if x == 0:
      minY = columnDepths[1]
    elif x == PLAYFIELD_WIDTH - 1:
      minY = columnDepths[PLAYFIELD_WIDTH - 2]
    else:
      minY = max(columnDepths[x - 1], columnDepths[x + 1])
    for y in range(columnDepths[x], minY - 1, -1):
      if ((x == 0 or playfield[y][x - 1] != NONE) and (x == PLAYFIELD_WIDTH - 1 
          or playfield[y][x + 1] != NONE)):
        e.wells += 1

  e.holes = 0
  e.columnTransitions = 0
  for x in range(PLAYFIELD_WIDTH):
    solid = True
    for y in range(columnDepths[x] + 1, PLAYFIELD_HEIGHT):
      if playfield[y][x] == NONE:
        if playfield[y - 1][x] != NONE:
          e.holes += 1
        if solid:
          solid = False
          e.columnTransitions += 1
      elif not solid:
        solid = True
        e.columnTransitions += 1

  e.rowTransitions = 0
  for y in range(PLAYFIELD_HEIGHT):
    solidFound = False
    solid = True
    transitions = 0
    for x in range(PLAYFIELD_WIDTH):
      if x == PLAYFIELD_WIDTH:
        if not solid:
          transitions += 1
      else:          
        if playfield[y][x] == NONE:            
          if solid:
            solid = False
            transitions += 1
        else:          
          solidFound = True
          if not solid:
            solid = True
            transitions += 1                           
    if solidFound:        
      e.rowTransitions += transitions

def clearRows(playfield, tetriminoY):
  rows = 0
  startRow = tetriminoY - 2
  endRow = tetriminoY + 1

  if startRow < 1:
    startRow = 1
  if endRow >= PLAYFIELD_HEIGHT:
    endRow = PLAYFIELD_HEIGHT - 1

  for y in range(startRow, endRow + 1):
    if playfield[y][PLAYFIELD_WIDTH] == PLAYFIELD_WIDTH:
      rows += 1
      clearRow(playfield, y)
  return rows

def clearRow(playfield, y):
  global spareIndex
  clearedRow = playfield[y]
  clearedRow[PLAYFIELD_WIDTH] = y
  for i in range(y, 0, -1):
    playfield[i] = playfield[i - 1]
  playfield[0] = spareRows[spareIndex]
  playfield[0][PLAYFIELD_WIDTH] = 0
  spareRows[spareIndex] = clearedRow
  spareIndex += 1
  
def restoreRow(playfield):
  global spareIndex
  spareIndex -= 1
  restoredRow = spareRows[spareIndex]
  y = restoredRow[PLAYFIELD_WIDTH]

  spareRows[spareIndex] = playfield[0]

  for i in range(y):
    playfield[i] = playfield[i + 1]
  restoredRow[PLAYFIELD_WIDTH] = PLAYFIELD_WIDTH
  playfield[y] = restoredRow

def restoreRows(playfield, rows):
  for i in range(rows):
    restoreRow(playfield)
    
globalMark = 1    
    
class Searcher(object):
  
  def __init__(self):
    self.states = [[[State(x, y, rotation) for rotation in range(4)] 
        for x in range(PLAYFIELD_WIDTH)] for y in range(PLAYFIELD_HEIGHT)]
    self.queue = []
  
  def lockPiece(self, playfield, tetriminoType, ID, state):
    for square in ORIENTATIONS[tetriminoType][state.rotation].squares:
      y = state.y + square.y
      if y >= 0:
        playfield[y][state.x + square.x] = tetriminoType
        playfield[y][PLAYFIELD_WIDTH] += 1
    searchListener(playfield, tetriminoType, ID, state)
    for square in ORIENTATIONS[tetriminoType][state.rotation].squares:
      y = state.y + square.y
      if y >= 0:
        playfield[y][state.x + square.x] = NONE
        playfield[y][PLAYFIELD_WIDTH] -= 1
  
  # returns true if the position is valid even if the node is not enqueued
  def addChild(self, playfield, tetriminoType, mark, state, x, y, rotation):
    orientation = ORIENTATIONS[tetriminoType][rotation]
    if x < orientation.minX or x > orientation.maxX or y > orientation.maxY:
      return False

    childNode = self.states[y][x][rotation]
    if childNode.visited == mark:
      return True

    for i in range(4):
      square = orientation.squares[i]
      playfieldY = y + square.y
      if playfieldY >= 0 and playfield[playfieldY][x + square.x] != NONE:
        return False

    childNode.visited = mark
    childNode.predecessor = state
        
    self.queue.insert(0, childNode)   
    return True 
  
  def search(self, playfield, tetriminoType, id):
    global globalMark
    
    maxRotation = len(ORIENTATIONS[tetriminoType]) - 1
    mark = globalMark
    globalMark += 1

    if not self.addChild(playfield, tetriminoType, mark, None, 5, 0, 0):
      return False

    while self.queue:
      state = self.queue.pop()
      
      if maxRotation != 0:
        self.addChild(playfield, tetriminoType, mark, state, state.x, state.y, 
            maxRotation if state.rotation == 0 else state.rotation - 1)
        if maxRotation != 1:
          self.addChild(playfield, tetriminoType, mark, state, state.x, state.y, 
              0 if state.rotation == maxRotation else state.rotation + 1)

      self.addChild(playfield, tetriminoType, mark, state, state.x - 1, state.y, 
          state.rotation)
      self.addChild(playfield, tetriminoType, mark, state, state.x + 1, state.y, 
          state.rotation)

      if not self.addChild(playfield, tetriminoType, mark, state, state.x, 
          state.y + 1, state.rotation):
        self.lockPiece(playfield, tetriminoType, id, state)

    return True
  
    
searchers = [Searcher() for i in range(TETRIMINOS_SEARCHED)]
tetriminoIndices = [None for i in range(TETRIMINOS_SEARCHED)]
e = PlayfieldEvaluation()
totalRows = 0
totalDropHeight = 0
bestFitness = 0.0
bestResult = None
result0 = None

def searchListener(playfield, tetriminoType, ID, state):
  global tetriminoIndices
  global bestFitness
  global bestResult
  global totalRows
  global totalDropHeight
  global result0

  if ID == 0:
    result0 = state

  orientation = ORIENTATIONS[tetriminoType][state.rotation]
  rows = clearRows(playfield, state.y)
  originalTotalRows = totalRows
  originalTotalDropHeight = totalDropHeight
  totalRows += rows
  totalDropHeight += orientation.maxY - state.y

  nextID = ID + 1

  if nextID == len(tetriminoIndices):
    evaluatePlayfield(playfield)
    fitness = computeFitness()
    if fitness < bestFitness:
      bestFitness = fitness
      bestResult = result0
  else:
    searchers[nextID].search(playfield, tetriminoIndices[nextID], nextID)

  totalDropHeight = originalTotalDropHeight
  totalRows = originalTotalRows
  restoreRows(playfield, rows)
  
def computeFitness():
  return (WEIGHTS[0] * totalRows
        + WEIGHTS[1] * totalDropHeight
        + WEIGHTS[2] * e.wells
        + WEIGHTS[3] * e.holes                 
        + WEIGHTS[4] * e.columnTransitions
        + WEIGHTS[5] * e.rowTransitions)
        
def search(playfield, indices):
  global tetriminoIndices
  global bestFitness
  global bestResult
  global searchers
  
  tetriminoIndices = indices
  bestResult = None
  bestFitness = sys.maxint
  searchers[0].search(playfield, tetriminoIndices[0], 0)
  return bestResult
 
def buildStatesList(state):
  s = state
  count = 0      
  while s != None:
    count += 1
    s = s.predecessor
  states = [None for i in range(count)]  
  while state != None:
    count -= 1
    states[count] = state
    state = state.predecessor
  return states

A = 0
B = 1
Select = 2
Start = 3
Up = 4
Down = 5
Left = 6
Right = 7

OrientationTable = 0x8A9C
TetriminoTypeTable = 0x993B
SpawnTable = 0x9956
Copyright1 = 0x00C3
Copyright2 = 0x00A8
GameState = 0x00C0
LowCounter = 0x00B1
HighCounter = 0x00B2
TetriminoX = 0x0060
TetriminoY1 = 0x0061
TetriminoY2 = 0x0041
TetriminoID = 0x0062
NextTetriminoID = 0x00BF
FallTimer = 0x0065
Playfield = 0x0400
Level = 0x0064
LevelTableAccess = 0x9808
LinesHigh = 0x0071
LinesLow = 0x0070
PlayState = 0x0068

EMPTY_SQUARE = 0xEF

nintaco.initRemoteAPI("localhost", 9999)
api = nintaco.getAPI()
tetriminos = [0 for i in range(TETRIMINOS_SEARCHED)]
playfield = createPlayfield()
TetriminosTypes = [0 for i in range(19)]
playFast = False
  
playingDelay = 0
targetTetriminoY = 0
startCounter = 0
movesIndex = 0
moving = False
states = None

def launch():
  api.addActivateListener(apiEnabled)
  api.addAccessPointListener(updateScore, nintaco.PreExecute, 0x9C35)
  api.addAccessPointListener(speedUpDrop, nintaco.PreExecute, 0x8977)
  api.addAccessPointListener(tetriminoYUpdated, nintaco.PreWrite, TetriminoY1)
  api.addAccessPointListener(tetriminoYUpdated, nintaco.PreWrite, TetriminoY2)
  api.addFrameListener(renderFinished)
  api.addStatusListener(statusChanged)
  api.run()

def apiEnabled():
  readTetriminoTypes()
  
def tetriminoYUpdated(accessPointType, address, tetriminoY):
  global targetTetriminoY
  if tetriminoY == 0:
    targetTetriminoY = 0
  if moving:      
    return targetTetriminoY
  else:
    return tetriminoY
  
def readTetriminoTypes():
  for i in range(19):
    TetriminosTypes[i] = api.readCPU(TetriminoTypeTable + i)
    
def resetPlayState(gameState):
  if gameState != 4:
    api.writeCPU(PlayState, 0)

def updateScore(accessPointType, address, value):
  # cap the points multiplier at 30 to avoid the kill screen
  if api.readCPU(0x00A8) > 30:
    api.writeCPU(0x00A8, 30)
  return -1

def speedUpDrop(accessPointType, address, value):
  api.setX(0x1E)
  return -1

def setTetriminoYAddress(address, y):
  global targetTetriminoY
  targetTetriminoY = y
  api.writeCPU(address, y)
  
def setTetriminoY(y):
  setTetriminoYAddress(TetriminoY1, y)
  setTetriminoYAddress(TetriminoY2, y)

def makeMove(tetriminoType, state, finalMove):
  if finalMove: 
    api.writeCPU(0x006E, 0x03)
  api.writeCPU(TetriminoX, state.x)
  setTetriminoY(state.y)
  api.writeCPU(TetriminoID, ORIENTATIONS[tetriminoType][state.rotation]
      .orientationID)

def readTetrimino():
  return TetriminosTypes[api.readCPU(TetriminoID)]

def readNextTetrimino():
  return TetriminosTypes[api.readCPU(NextTetriminoID)]

def readPlayfield():
  tetriminos[0] = readTetrimino()
  tetriminos[1] = readNextTetrimino()

  for i in range(PLAYFIELD_HEIGHT):
    playfield[i][10] = 0
    for j in range(PLAYFIELD_WIDTH):
      if api.readCPU(Playfield + 10 * i + j) == EMPTY_SQUARE:
        playfield[i][j] = NONE
      else:
        playfield[i][j] = I
        playfield[i][10] += 1

def spawned():
  currentTetrimino = api.readCPU(TetriminoID)
  playState = api.readCPU(PlayState)
  tetriminoX = api.readCPU(TetriminoX)
  tetriminoY = api.readCPU(TetriminoY1)

  return (playState == 1 and tetriminoX == 5 and tetriminoY == 0 
      and currentTetrimino < len(TetriminosTypes))

def isPlaying(gameState):
  return gameState == 4 and api.readCPU(PlayState) < 9

def pressStart():
  global startCounter
  if startCounter > 0:
    startCounter -= 1
  else:
    startCounter = 10
  if startCounter >= 5:
    api.writeGamepad(0, Start, True)

def skipCopyrightScreen(gameState):
  if gameState == 0:
    if api.readCPU(Copyright1) > 1:
      api.writeCPU(Copyright1, 0)
    elif api.readCPU(Copyright2) > 2:
      api.writeCPU(Copyright2, 1)

def skipTitleAndDemoScreens(gameState):
  global startCounter
  if gameState == 1 or gameState == 5:
    pressStart()   
  else:
    startCounter = 0
    
def renderFinished():
  global moving
  global movesIndex
  global states
  global playingDelay
  
  gameState = api.readCPU(GameState)
  skipCopyrightScreen(gameState)
  skipTitleAndDemoScreens(gameState)
  resetPlayState(gameState)

  if isPlaying(gameState):
    if playingDelay > 0:
      playingDelay -= 1
    elif playFast:
      # skip line clearing animation
      if api.readCPU(PlayState) == 4:
        api.writeCPU(PlayState, 5)
      if spawned():
        readPlayfield()
        state = search(playfield, tetriminos)
        if state != None:
          moving = True
          makeMove(tetriminos[0], state, True)
          moving = False
    else:
      if moving and movesIndex < len(states):
        makeMove(tetriminos[0], states[movesIndex], 
            movesIndex == len(states) - 1)
        movesIndex += 1
      else:          
        moving = False
        if spawned():
          readPlayfield()
          state = search(playfield, tetriminos)
          if state != None:
            states = buildStatesList(state)
            movesIndex = 0
            moving = True
  else:
    states = None
    moving = False
    playingDelay = 16

def statusChanged(message):
  print(message)

def main(args):
  global playFast
  playFast = len(args) > 1 and "fast" == args[1].lower()
  launch()

if __name__ == "__main__":
  main(sys.argv)

