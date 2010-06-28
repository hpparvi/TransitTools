import time
import curses
stdscr = curses.initscr()
curses.noecho()
curses.cbreak()
stdscr.keypad(1)

pad = curses.newpad(50, 50)
for y in range(0, 10):
    for x in range(0, 50):
        try: pad.addch(y,x, ord('a') + (x*x+y*y) % 26 )
        except curses.error: pass

    pad.refresh( 0,0, 5,5, 20,75)
    time.sleep(1)

curses.nocbreak(); stdscr.keypad(0); curses.echo()
curses.endwin()
