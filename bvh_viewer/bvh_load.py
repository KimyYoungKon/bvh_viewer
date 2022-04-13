import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
import argparse

from mpl_toolkits.mplot3d import axes3d, Axes3D
from numpy.core.fromnumeric import shape
from transforms3d.euler import euler2mat, mat2euler


class Motion:
    def __init__(self, name, parent, index):
        self.name = name
        self.parent = parent
        self.offset = np.zeros(3)
        self.children = []
        self.channels = []
        self.motion = []
        self.index = index


class Skeleton:
    def __init__(self):
        self.joints = []
        self.bones = []
        self.root = None
        self.frameMatrix = []
        self.frameTime = 0
        self.frameNum = 0

    def parseInit(self, lines):
        jointStack = []
        cnt = 0
        for line in lines:
            if line == '':
                continue
            words = re.split(r'\s+', line)
            if words[0] == "ROOT":
                joint = Motion(words[1], None, cnt)
                jointStack.append(joint)
                self.root = joint
                self.joints.append(joint)
                cnt += 1
            elif words[0] == "JOINT":
                parent = jointStack[-1]
                joint = Motion(words[1], parent, cnt)
                parent.children.append(joint)
                jointStack.append(joint)
                self.joints.append(joint)
                self.bones.append(list([parent.index, joint.index]))
                cnt += 1
            elif words[0] == "CHANNELS":
                for i in range(2, len(words)):
                    jointStack[-1].channels.append(words[i])
            elif words[0] == "OFFSET":
                for i in range(1, len(words)):
                    jointStack[-1].offset[i - 1] = float(words[i])
            elif words[0] == "End":
                joint = Motion("End", jointStack[-1], cnt)
                jointStack[-1].children.append(joint)
                self.bones.append(list([jointStack[-1].index, joint.index]))
                jointStack.append(joint)
                self.joints.append(joint)
                cnt += 1
            elif words[0] == "}":
                jointStack.pop()

    def parseMotion(self, lines):
        for line in lines:
            if line == '':
                continue
            words = re.split(r'\s+', line)
            if line.startswith("Frame Time"):
                self.frameTime = 1000 * float(words[2])
                continue
            if line.startswith("Frames"):
                self.frameNum = int(words[1])
                continue

            curFrame = []
            for word in words:
                curFrame.append(float(word))
            self.frameMatrix.append(curFrame)

        # numpy array
        self.frameMatrix = np.array(self.frameMatrix)


    def parse(self, lines):
        lines = [line.strip() for line in lines]
        l = lines.index("MOTION")
        init = lines[:l]
        motion = lines[l + 1:]
        self.parseInit(init)
        self.parseMotion(motion)


    def getAllPos(self):
        global motionIndex
        for i in range(self.frameNum):
            motionIndex = 0
            self.getJointPos(i, self.root, np.eye(3), np.zeros(3))


    def getJointPos(self, frameIndex, joint, parentMatrix, parentPos):
        global motionIndex

        offset = np.zeros(3)
        curMatrix = np.eye(3)
        for i in joint.channels:
            if i == "Xrotation":
                rotation = self.frameMatrix[frameIndex, motionIndex]
                rotation = np.deg2rad(rotation)
                eulerRot = np.array([rotation, 0., 0.])
                M = euler2mat(*eulerRot)
                curMatrix = curMatrix.dot(M)
            elif i == "Yrotation":
                rotation = self.frameMatrix[frameIndex, motionIndex]
                rotation = np.deg2rad(rotation)
                eulerRot = np.array([0., rotation, 0.])
                M = euler2mat(*eulerRot)
                curMatrix = curMatrix.dot(M)
            elif i == "Zrotation":
                rotation = self.frameMatrix[frameIndex, motionIndex]
                rotation = np.deg2rad(rotation)
                eulerRot = np.array([0., 0., rotation])
                M = euler2mat(*eulerRot)
                curMatrix = curMatrix.dot(M)
            elif i == "Xposition":
                offset[0] = self.frameMatrix[frameIndex, motionIndex]
            elif i == "Yposition":
                offset[1] = self.frameMatrix[frameIndex, motionIndex]
            elif i == "Zposition":
                offset[2] = self.frameMatrix[frameIndex, motionIndex]
            motionIndex += 1


        newPos = parentPos + parentMatrix.dot(joint.offset) + offset
        newMatrix = parentMatrix.dot(curMatrix)
        joint.motion.append(newPos)


        for c in joint.children:
            self.getJointPos(frameIndex, c, newMatrix, newPos)
        return



def update(num, lines, boneDatas):
    for line, data in zip(lines, boneDatas):
        line.set_data([data[:, num, 0], data[:, num, 2]])
        line.set_3d_properties(data[:, num, 1])
    return lines


if __name__ == '__main__':
    # args = parser.parse_args()
    # bvhFile=args.bvhFile;
    bvhFile = 'dance.bvh';
    with open(bvhFile, 'r') as fin:
        lines = fin.readlines()


    sk = Skeleton()
    sk.parse(lines)


    motionIndex = 0

    sk.getAllPos()
    boneDatas = []
    for bone in sk.bones:
        jointA = bone[0]
        jointB = bone[1]
        posA = sk.joints[jointA].motion
        posB = sk.joints[jointB].motion
        boneDatas.append([posA, posB])

    boneDatas = np.array(boneDatas)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.set_zlim(-1, 79)
    plt.axis('off')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    lines = [ax.plot([boneData[0, 0, 0], boneData[1, 0, 0]], [boneData[0, 0, 2], boneData[1, 0, 2]],
                     [boneData[0, 0, 1], boneData[1, 0, 1]])[0] for boneData in boneDatas]

    ani = animation.FuncAnimation(fig, update,
                                  sk.frameNum, interval=sk.frameTime, blit=True, fargs=(lines, boneDatas))

    plt.show()

    # ani.save('motion.mp4')

    print("done.")

