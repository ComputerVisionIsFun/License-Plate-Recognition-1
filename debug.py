import cv2




def highlight(senstance, no='93'):
    '''
    https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal-in-python
    '''
    # print('\n \033[91m' +'='*10+ senstance + '='*10+'\033[0m')    
    print('\n \033[' + no + 'm' +'='*10+ ' '+ senstance + ' ' + '='*10 + '\033[0m')   

def make_grid(img, xpts = [], ypts = []):
    rows, cols, _ = img.shape
    # print(rows, cols)
    output = img.copy()
    color = (100,100,100)
    # vertical lines
    for xpt in xpts:
        x = int(xpt)
        cv2.line(output, (x, 0), (x, rows), color, 1, 8, 0)
        
    # horzinontal lines
    for ypt in ypts:
        y = int(ypt)
        cv2.line(output, (0, y), (cols, y), color, 1, 8, 0)
    
    return output


def draw_rectangle(img, cx, cy, width, height, color = (0,255,0)):
    xmin, ymin = int(cx - width/2), int(cy - height/2)
    xmax, ymax = int(cx + width/2), int(cy + height/2)
    cv2.rectangle(img, (xmin,ymin),(xmax,ymax),color,3,8,0)
    return img
