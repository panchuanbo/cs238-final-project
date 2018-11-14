#############################################
# This allows users to enter custom inputs. #
# Useful for testing out certain actions.   #
#############################################

import sys

from nintaco import nintaco
from threading import Thread

actions = {
  0: 'A',
  1: 'B',
  2: 'Select',
  3: 'Start',
  4: 'Up',
  5: 'Down',
  6: 'Left',
  7: 'Right'
}

nintaco.initRemoteAPI("localhost", 9999)
api = nintaco.getAPI()

def launch():
  api.addActivateListener(apiEnabled)
  api.run()

def apiEnabled():
  thread = Thread(target=moveHandler)
  thread.daemon = True
  thread.start()

def moveHandler():
  while True:
    move = int(raw_input("Please Enter a Move: "))
    api.writeGamepad(0, move, True)

def main(args):
  print 'Buttons -> Values'
  for (k, v) in actions.iteritems():
    print '\t%s -> %s' % (k, v)

  launch()

if __name__ == "__main__":
  main(sys.argv)

