import pygame
from pygame.locals import *
import random
import torch
from torch import tensor, float16, Tensor
import math
import time

# particls in sim
k_particlesPerCreature = 1024

# define colors
red = (255, 0, 0)
blue = (0, 0, 255)
green = (0, 255, 0)
yellow = (255, 255, 0)
black = (0, 0, 0)

# init pygame
pygame.init()
width = 800
height = 800
screen = pygame.display.set_mode((width + 1, height + 1))
pygame.display.set_caption("Particle Life")
screen.fill(black)

clock = pygame.time.Clock()

# set up device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
if torch.backends.mps.is_available():
    print("running on MPS")
elif torch.cuda.is_available():
    print("running on CUDA")
else:
    print("running on cpu")


class ParticleLife():
    
    def __init__(self, size: int) -> None:
        # create tensors for position on device
        self.PositionRedGpu = tensor([[random.randint(0, width) for x in range(math.floor(size/4))], [random.randint(0, height) for x in range(math.floor(size/4))]], dtype=int, device=device)
        self.PositionBlueGpu = tensor([[random.randint(0, width) for x in range(math.floor(size/4))], [random.randint(0, height) for x in range(math.floor(size/4))]], dtype=int, device=device)
        self.PositionGreenGpu = tensor([[random.randint(0, width) for x in range(math.floor(size/4))], [random.randint(0, height) for x in range(math.floor(size/4))]], dtype=int, device=device)
        self.PositionYellowGpu = tensor([[random.randint(0, width) for x in range(math.floor(size/4))], [random.randint(0, height) for x in range(math.floor(size/4))]], dtype=int, device=device)
        if device != "cpu":
            # create tensors for position on cpu to allow for easier drawing if not running on cpu
            self.PositionRed = self.PositionRedGpu.to("cpu")
            self.PositionBlue = self.PositionBlueGpu.to("cpu")
            self.PositionGreen = self.PositionGreenGpu.to("cpu")
            self.PositionYellow = self.PositionYellowGpu.to("cpu")
            
            # create numpy array that shares memory with the cpu tensors for easier drawing
            self.PositionRedArray = self.PositionRed.numpy()
            self.PositionBlueArray = self.PositionBlue.numpy()
            self.PositionGreenArray = self.PositionGreen.numpy()
            self.PositionYellowArray = self.PositionYellow.numpy()
        else:
            self.PositionRedArray = self.PositionRedGpu.numpy()
            self.PositionBlueArray = self.PositionBlueGpu.numpy()
            self.PositionGreenArray = self.PositionGreenGpu.numpy()
            self.PositionYellowArray = self.PositionYellowGpu.numpy()
        
        # create tensors for velocity on device
        self.VelocityRed = torch.zeros((2, math.floor(size/4)), dtype=float16, device=device)
        self.VelocityBlue = torch.zeros((2, math.floor(size/4)), dtype=float16, device=device)
        self.VelocityGreen = torch.zeros((2, math.floor(size/4)), dtype=float16, device=device)
        self.VelocityYellow = torch.zeros((2, math.floor(size/4)), dtype=float16, device=device)
        
        # rules
        self.RedRed = random.randint(-100, 100)/(size*2)
        self.RedBlue = random.randint(-100, 100)/(size*2)
        self.RedGreen = random.randint(-100, 100)/(size*2)
        self.RedYellow = random.randint(-100, 100)/(size*2)
        self.BlueRed = random.randint(-100, 100)/(size*2)
        self.BlueBlue = random.randint(-100, 100)/(size*2)
        self.BlueGreen = random.randint(-100, 100)/(size*2)
        self.BlueYellow = random.randint(-100, 100)/(size*2)
        self.GreenRed = random.randint(-100, 100)/(size*2)
        self.GreenBlue = random.randint(-100, 100)/(size*2)
        self.GreenGreen = random.randint(-100, 100)/(size*2)
        self.GreenYellow = random.randint(-100, 100)/(size*2)
        self.YellowRed = random.randint(-100, 100)/(size*2)
        self.YellowBlue = random.randint(-100, 100)/(size*2)
        self.YellowGreen = random.randint(-100, 100)/(size*2)
        self.YellowYellow = random.randint(-100, 100)/(size*2)
        
    # sync cpu tensor values with gpu tensor values
    def syncCpuTensors(self):
        if device == "cpu":
            return
        self.PositionRed.copy_(self.PositionRedGpu)
        self.PositionBlue.copy_(self.PositionBlueGpu)
        self.PositionGreen.copy_(self.PositionGreenGpu)
        self.PositionYellow.copy_(self.PositionYellowGpu)

    def calculateVelocity(self, tensor1: Tensor, vTensor: Tensor, tensor2: Tensor, rule: float):
        # create a tensor from tensor1 repeated for each value in tensor2 to allow all calculations to be done at once
        PositionTensor1 = tensor1.repeat((1, tensor2.size(dim=1)))
        # create a tensor from tensor2 repeat_interleaved for each value in tensor1 to allow all calulations to be done at once
        PositionTensor2 = torch.repeat_interleave(tensor2, tensor1.size(dim=1), dim=1)
        # create a tensor to calculate the force on each particle
        forceTensor = torch.zeros((2, PositionTensor1.size(dim=1)), device=device)
        
        # calculate distance along x, y, and xy directions
        disXY = torch.abs(PositionTensor1.sub(PositionTensor2))
        dis = (torch.sum(disXY**2, dim=0))**.5
        
        # calculate forces
        F = torch.reciprocal((dis + .0001)) * rule
        forceTensor += torch.stack([F, F]).multiply(disXY)
        
        # reshape force Tensor to match with the position tensor, then add along the z axis
        forceTensor = torch.stack((forceTensor[0].reshape(tensor2.size(dim=1), tensor1.size(dim=1)), forceTensor[1].reshape(tensor2.size(dim=1), tensor1.size(dim=1))), dim=1)
        forceTensor = forceTensor.sum(dim=0)
        
        # update velocity tensor
        vTensor.copy_((vTensor + forceTensor)*0.5)
        
    def updatePosition(self):
        # velocity to position
        self.PositionRedGpu.add_(self.VelocityRed.long())
        self.PositionBlueGpu.add_(self.VelocityBlue.long())
        self.PositionGreenGpu.add_(self.VelocityGreen.long())
        self.PositionYellowGpu.add_(self.VelocityYellow.long())
        
        self.PositionRedGpu[0] = self.PositionRedGpu[0].clamp(0, width)
        self.PositionRedGpu[1] = self.PositionRedGpu[1].clamp(0, height)
        self.PositionBlueGpu[0] = self.PositionBlueGpu[0].clamp(0, width)
        self.PositionBlueGpu[1] = self.PositionBlueGpu[1].clamp(0, height)
        self.PositionGreenGpu[0] = self.PositionGreenGpu[0].clamp(0, width)
        self.PositionGreenGpu[1] = self.PositionGreenGpu[1].clamp(0, height)
        self.PositionYellowGpu[0] = self.PositionYellowGpu[0].clamp(0, width)
        self.PositionYellowGpu[1] = self.PositionYellowGpu[1].clamp(0, height)
        
        # reverse velocity if object is outside of the bounds
        self.VelocityRed[0].multiply_(torch.where((self.PositionRedGpu[0] == width), -1.0, 1.0))
        self.VelocityBlue[0].multiply_(torch.where((self.PositionBlueGpu[0] == width), -1.0, 1.0))
        self.VelocityGreen[0].multiply_(torch.where((self.PositionGreenGpu[0] == width), -1.0, 1.0))
        self.VelocityYellow[0].multiply_(torch.where((self.PositionYellowGpu[0] == width), -1.0, 1.0))
        self.VelocityRed[0].multiply_(torch.where((self.PositionRedGpu[0] == 0), -1.0, 1.0))
        self.VelocityBlue[0].multiply_(torch.where((self.PositionBlueGpu[0] == 0), -1.0, 1.0))
        self.VelocityGreen[0].multiply_(torch.where((self.PositionGreenGpu[0] == 0), -1.0, 1.0))
        self.VelocityYellow[0].multiply_(torch.where((self.PositionYellowGpu[0] == 0), -1.0, 1.0))
        
        self.VelocityRed[1].multiply_(torch.where((self.PositionRedGpu[1] == height), -1.0, 1.0))
        self.VelocityBlue[1].multiply_(torch.where((self.PositionBlueGpu[1] == height), -1.0, 1.0))
        self.VelocityGreen[1].multiply_(torch.where((self.PositionGreenGpu[1] == height), -1.0, 1.0))
        self.VelocityYellow[1].multiply_(torch.where((self.PositionYellowGpu[1] == height), -1.0, 1.0))
        self.VelocityRed[1].multiply_(torch.where((self.PositionRedGpu[1] == 0), -1.0, 1.0))
        self.VelocityBlue[1].multiply_(torch.where((self.PositionBlueGpu[1] == 0), -1.0, 1.0))
        self.VelocityGreen[1].multiply_(torch.where((self.PositionGreenGpu[1] == 0), -1.0, 1.0))
        self.VelocityYellow[1].multiply_(torch.where((self.PositionYellowGpu[1] == 0), -1.0, 1.0))
        
    # draw particles
    def draw(self):
        for i in range(len(self.PositionRedArray[0])):
            pygame.draw.rect(screen, red, (self.PositionRedArray[0][i] - 1, self.PositionRedArray[1][i] - 1, 3, 3))
        for i in range(len(self.PositionBlueArray[0])):
            pygame.draw.rect(screen, blue, (self.PositionBlueArray[0][i] - 1, self.PositionBlueArray[1][i] - 1, 3, 3))
        for i in range(len(self.PositionGreenArray[0])):
            pygame.draw.rect(screen, green, (self.PositionGreenArray[0][i] - 1, self.PositionGreenArray[1][i] - 1, 3, 3))
        for i in range(len(self.PositionYellowArray[0])):
            pygame.draw.rect(screen, yellow, (self.PositionYellowArray[0][i] - 1, self.PositionYellowArray[1][i] - 1, 3, 3))
        
    # main loop for a particle sim
    def update(self):
        self.calculateVelocity(self.PositionRedGpu, self.VelocityRed, self.PositionRedGpu, self.RedRed)
        self.calculateVelocity(self.PositionRedGpu, self.VelocityRed, self.PositionBlueGpu, self.RedBlue)
        self.calculateVelocity(self.PositionRedGpu, self.VelocityRed, self.PositionGreenGpu, self.RedGreen)
        self.calculateVelocity(self.PositionRedGpu, self.VelocityRed, self.PositionYellowGpu, self.RedYellow)
        self.calculateVelocity(self.PositionBlueGpu, self.VelocityBlue, self.PositionRedGpu, self.BlueRed)
        self.calculateVelocity(self.PositionBlueGpu, self.VelocityBlue, self.PositionBlueGpu, self.BlueBlue)
        self.calculateVelocity(self.PositionBlueGpu, self.VelocityBlue, self.PositionGreenGpu, self.BlueGreen)
        self.calculateVelocity(self.PositionBlueGpu, self.VelocityBlue, self.PositionYellowGpu, self.BlueYellow)
        self.calculateVelocity(self.PositionGreenGpu, self.VelocityGreen, self.PositionRedGpu, self.GreenRed)
        self.calculateVelocity(self.PositionGreenGpu, self.VelocityGreen, self.PositionBlueGpu, self.GreenBlue)
        self.calculateVelocity(self.PositionGreenGpu, self.VelocityGreen, self.PositionGreenGpu, self.GreenGreen)
        self.calculateVelocity(self.PositionGreenGpu, self.VelocityGreen, self.PositionYellowGpu, self.GreenYellow)
        self.calculateVelocity(self.PositionYellowGpu, self.VelocityYellow, self.PositionRedGpu, self.YellowRed)
        self.calculateVelocity(self.PositionYellowGpu, self.VelocityYellow, self.PositionBlueGpu, self.YellowBlue)
        self.calculateVelocity(self.PositionYellowGpu, self.VelocityYellow, self.PositionGreenGpu, self.YellowGreen)
        self.calculateVelocity(self.PositionYellowGpu, self.VelocityYellow, self.PositionYellowGpu, self.YellowYellow)

        self.updatePosition()

        self.syncCpuTensors()

        self.draw()

def main():
    running = True
    particleLife = ParticleLife(k_particlesPerCreature)
    timeList = []
    while running:
        start = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill(black)
        
        particleLife.update()

        pygame.display.update()
        end = time.time()
        timeList.append(end-start)
        clock.tick(120)
    print("average time: " + str(sum(timeList)/len(timeList)))
    pygame.quit()
        
if __name__ == "__main__":
    main()