import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import OpenGL.GL.shaders as shaders
import math
import numpy as np
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

KEY_S = False # smoothing
KEY_Z = True # solid mode
KEY_B = False # box body mode
KEY_SPACE = False # for animation


orbit = False
panning = False

Azimuth = 0
Elevation = 30
distance = 5

prev_xpos = -1
prev_ypos = -1
pan_lr = 0
pan_ud = 0
distance = 5

# for bvh
frame_time = None
frame_num = None
motion = []
hierarchy = None
ON = False
frame_now = 0
scale = 1

def render():
    global Azimuth, Elevation, distance
    global pan_lr, pan_ud
    global ON, KEY_Z
    global hierarchy
    global glScale

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)

    if KEY_Z:
        #solid
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    else:
        #wire frame
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glFrustum(-1,1, -1,1,1,10)

    A=np.radians(Azimuth)
    E=np.radians(Elevation)
    d=distance

    eyeX= d*np.cos(E)*np.sin(A) + pan_lr*np.cos(A)
    eyeY= d*np.sin(E) + pan_ud
    eyeZ= d*np.cos(E)*np.cos(A) - pan_lr*np.sin(A)
    atX= - pan_lr*np.cos(A)
    atY= pan_ud
    atZ= - pan_lr*np.sin(A)
    upX= 0
    upY= np.cos(E)
    upZ= 0

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(eyeX,eyeY,eyeZ,atX,atY,atZ,upX,upY,upZ)
    drawFrame()

    if ON: #obj file
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_RESCALE_NORMAL)

        lightPos = (-5,4,-3,1)
        glLightfv(GL_LIGHT0, GL_POSITION, lightPos)
        lightPos = (3,4,5,1)
        glLightfv(GL_LIGHT1, GL_POSITION, lightPos)

        #light
        ambientLightColor = (0.1,0.1,0.1,1)
        lightColor=(1,1,1,1)
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLightColor)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, lightColor)
        glLightfv(GL_LIGHT0, GL_SPECULAR, lightColor)
        glLightfv(GL_LIGHT1, GL_AMBIENT, ambientLightColor)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, lightColor)
        glLightfv(GL_LIGHT1, GL_SPECULAR, lightColor)

        #material
        specularObjectColor = (0.9,0.9,0.9,1)
        objectColor = (0,0,1,1)
        glMaterialfv(GL_FRONT, GL_SPECULAR, specularObjectColor)
        glMaterialfv(GL_FRONT, GL_AMBIENT, objectColor)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, objectColor)
        glMaterialfv(GL_FRONT, GL_SHININESS, 10)
        glScalef(scale, scale, scale)
        drawObj(hierarchy)
        glDisable(GL_LIGHTING)


def drawObj(data):
    global motion, KEY_SPACE, frame_now, frame_num, KEY_B

    tmp=data
    for i in range(len(tmp.children)):
        glPushMatrix()
        glTranslatef(tmp.offset[0], tmp.offset[1], tmp.offset[2])
        if KEY_SPACE:
            if tmp.position != None:
                for axis,col in tmp.position:
                    val = motion[frame_now][col]
                    if axis == 'x':
                        glTranslatef(val, 0, 0)
                    elif axis == 'y':
                        glTranslatef(0,val,0)
                    elif axis == 'z':
                        glTranslatef(0,0,val)
            if tmp.rotation != None:
                for axis,col in tmp.rotation:
                    val = motion[frame_now][col]
                    if axis == 'x':
                        glRotatef(val,1,0,0)
                    elif axis == 'y':
                        glRotatef(val,0,1,0)
                    elif axis == 'z':
                        glRotatef(val,0,0,1)

            if KEY_B:
                drawBox(tmp.children[i].offset)
            else:
                drawLine(tmp.children[i].offset)
            drawObj(tmp.children[i])
            glPopMatrix()

def drawLine(p):
    glBegin(GL_LINES)
    glVertex3f(0,0,0)
    glVertex3f(p[0],p[1],p[2])
    glEnd()


def drawBox(p):
    global scale
    x,y,z=p
    l = math.sqrt(x**2+y**2+z**2)
    if l == 0:
        return
    if x == 0 and z == 0:
        s=1
        c=0
    else:
        s=z/math.sqrt(x**2+z**2)
        c=x/math.sqrt(x**2+z**2)
    R1 = np.identity(4)
    R1[:3,:3] = [[c,0.,-s],
                 [0.,1.,0.],
                 [s,0.,c]]
    glPushMatrix()
    glMultMatrixf(R1.T)
    s=y/l
    c=math.sqrt(x**2+z**2)/l
    R2=np.identity(4)
    R2[:3,:3] = [[c, -s, 0.],
                 [s,c,0.],
                 [0.,0.,1.]]
    glMultMatrixf(R2.T)
    glScalef(l*.7,.04,.04)
    drawCube()
    glPopMatrix()

def drawCube():
    glBegin(GL_QUADS)

    glVertex3f(1, .5, -.5)
    glVertex3f(0, .5, -.5)
    glVertex3f(0, .5, .5)
    glVertex3f(1, .5, .5)

    glVertex3f(1, -.5, .5)
    glVertex3f(0, -.5, .5)
    glVertex3f(0, -.5, -.5)
    glVertex3f(1, -.5, -.5)

    glVertex3f(1, .5, .5)
    glVertex3f(0, .5, .5)
    glVertex3f(0, -.5, .5)
    glVertex3f(1, -.5, .5)

    glVertex3f(1, -.5, -.5)
    glVertex3f(0, -.5, -.5)
    glVertex3f(0, .5, -.5)
    glVertex3f(1, .5, -.5)

    glVertex3f(0, .5, .5)
    glVertex3f(0, .5, -.5)
    glVertex3f(0, -.5, -.5)
    glVertex3f(0, -.5, .5)

    glVertex3f(1, .5, -.5)
    glVertex3f(1, .5, .5)
    glVertex3f(1, -.5, .5)
    glVertex3f(1, -.5, -.5)

    glEnd()


def drawFrame():
    glBegin(GL_LINES)
    glColor3ub(255, 0, 0)
    glVertex3fv(np.array([-5.,0.,0.]))
    glVertex3fv(np.array([5.,0.,0.]))
    glColor3ub(50,50,50)
    for i in np.linspace(-5,5,30):
        glVertex3fv(np.array([-5.,0.,i]))
        glVertex3fv(np.array([5.,0.,i]))
    for i in np.linspace(-5,5,30):
        glVertex3fv(np.array([i,0.,-5.]))
        glVertex3fv(np.array([i,0.,5.]))
    glColor3ub(0, 255, 0)
    glVertex3fv(np.array([0.,-3.,0.]))
    glVertex3fv(np.array([0.,3.,0.]))
    glColor3ub(0, 0, 255)
    glVertex3fv(np.array([0.,0.,-5.]))
    glVertex3fv(np.array([0.,0.,5.]))
    glEnd()


def drop_callback(window, paths):
    logger.warning("drop callback")
    global motion, hierarchy, ON, KEY_SPACE, frame_now, frame_time, frame_num
    motion = []
    hierarchy = None
    ON = False
    KEY_SPACE = False
    frame_now = 0
    frame_time = None
    frame_num = None


def handle_dropped_file(path):
    global ON
    global frame_time, frame_num
    global motion, hierarchy
    global scale

    file=open(path, 'r')
    col = 0
    mode = 0
    now = None
    joints = []
    hierarchy_data = None
    motion_data = []
    scale = 1

    while True:
        line = file.readline()
        if not line:
            break
        tmp = line.strip("\n").split()

        logger.info("=======================================")
        logger.info(tmp)
        logger.info("=======================================")



def button_callback(window,button,action,mod):
    logger.warning("button click")
    global orbit, panning, prev_xpos, prev_ypos
    if button==glfw.MOUSE_BUTTON_LEFT:
        if action==glfw.PRESS:
            orbit = True
            prev_xpos = -1
            prev_ypos = -1
        elif action==glfw.RELEASE:
            orbit = False
        if button==glfw.MOUSE_BUTTON_RIGHT:
            if action==glfw.PRESS:
                panning=True
                prev_xpos = -1
                prev_ypos = -1
            elif action==glfw.RELEASE:
                panning = False

def scroll_callback(window,xoffset,yoffset):
    logger.warning("scroll callback")
    global distance
    if yoffset == -1 and distance<1000:
        distance += 0.5
    elif yoffset == 1 and distance>0.5:
        distance -= 0.5


def cursor_callback(window, xpos, ypos):
    logger.warning("cursor callback")
    global prev_xpos, prev_ypos, Azimuth, Elevation, orbit, panning, pan_lr, pan_ud
    if orbit:
        if prev_xpos != -1 and prev_ypos != -1:
            xdif = xpos - prev_xpos
            ydif = ypos - prev_ypos
            if Elevation < 90 or Elevation >= 270:
                Azimuth -= xdif*2
                Elevation += ydif*2
            elif Elevation >= 90 and Elevation < 270:
                Azimuth += xdif*2
                Elevation += ydif*2
            Azimuth = Azimuth%360
            Elevation = Elevation%360
        prev_xpos = xpos
        prev_ypos = ypos
    elif panning:
        if prev_xpos != -1 and prev_ypos != -1:
            xdif = xpos - prev_xpos
            ydif = ypos - prev_ypos
            if Elevation < 180:
                pan_lr -= xdif/200
                pan_ud += ydif/200
            elif Elevation >= 180:
                pan_lr -= xdif/200
                pan_ud -= ydif/200
        prev_xpos=xpos
        prev_ypos=ypos


def main():
  if not glfw.init():
      return


  window = glfw.create_window(800, 600, "My OpenGL window", None, None)

  if not window:
      glfw.terminate()
      return

  glfw.make_context_current(window)
  glfw.set_drop_callback(window, drop_callback)
  glfw.set_mouse_button_callback(window, button_callback)
  glfw.set_cursor_pos_callback(window,cursor_callback)
  glfw.set_scroll_callback(window,scroll_callback)


  glfw.swap_interval(1)
  while not glfw.window_should_close(window):
      glfw.poll_events()
      render()
      glfw.swap_buffers(window)
      if KEY_SPACE:
          frame_now += 1
          frame_now %= frame_num

  glfw.terminate()



if __name__ == "__main__":
    main()