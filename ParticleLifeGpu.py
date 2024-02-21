import pygame
from pygame.locals import *
import random
import torch
from torch import tensor, float16, Tensor
import math
import time

# particles in sim
k_particlesPerCreature = 4096

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
        
        # create tensors to store force values that will be added up at the end of the calculate velocity iterations
        self.ForceRedStored = torch.zeros((1, 2, math.floor(size/4)), dtype=float16, device=device).to(float16)
        self.ForceBlueStored = torch.zeros((1, 2, math.floor(size/4)), dtype=float16, device=device).to(float16)
        self.ForceGreenStored = torch.zeros((1, 2, math.floor(size/4)), dtype=float16, device=device).to(float16)
        self.ForceYellowStored = torch.zeros((1, 2, math.floor(size/4)), dtype=float16, device=device).to(float16)
        
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

    def calculateForces(self, tensor1: Tensor, forceStoredTensor: Tensor, tensor2: Tensor, rule: float):
        # create a tensor from tensor1 repeated for each value in tensor2 to allow all calculations to be done at once
        PositionTensor1 = tensor1.repeat((1, tensor2.size(dim=1)))
        # create a tensor from tensor2 repeat_interleaved for each value in tensor1 to allow all calulations to be done at once
        PositionTensor2 = torch.repeat_interleave(tensor2, tensor1.size(dim=1), dim=1)
        # create a tensor to calculate the force on each particle
        forceTensor = torch.zeros((2, PositionTensor1.size(dim=1)), dtype=float16, device=device)
        
        # calculate distance along x, y, and xy directions
        disXY = PositionTensor1.sub(PositionTensor2)
        dis = (torch.sum(disXY**2, dim=0))**.5
        
        # calculate forces
        F = torch.reciprocal((dis + .0001)) * rule
        forceTensor += torch.stack([F, F]).multiply(disXY)
        
        # reshape force Tensor to match with the position tensor, then add along the z axis
        forceTensor = torch.stack((forceTensor[0].reshape(tensor2.size(dim=1), tensor1.size(dim=1)), forceTensor[1].reshape(tensor2.size(dim=1), tensor1.size(dim=1))), dim=1)
        forceTensor = forceTensor.sum(dim=0)
        
        # store forces in a tensor to be added up later
        forceStoredTensor.set_(torch.cat([forceStoredTensor, forceTensor.unsqueeze(0)], dim=0))
        
    def updatePosition(self):
        # add up force stored tensors
        self.ForceRedStored = torch.sum(self.ForceRedStored, dim=0)
        self.ForceBlueStored = torch.sum(self.ForceBlueStored, dim=0)
        self.ForceGreenStored = torch.sum(self.ForceGreenStored, dim=0)
        self.ForceYellowStored = torch.sum(self.ForceYellowStored, dim=0)
        # forces to velocity
        self.VelocityRed.copy_((self.VelocityRed + self.ForceRedStored)*0.5)
        self.VelocityBlue.copy_((self.VelocityBlue + self.ForceBlueStored)*0.5)
        self.VelocityGreen.copy_((self.VelocityGreen + self.ForceGreenStored)*0.5)
        self.VelocityYellow.copy_((self.VelocityYellow + self.ForceYellowStored)*0.5)
        # reset force stored tensors
        self.ForceRedStored = torch.zeros((1, 2, math.floor(self.VelocityRed.size(dim=1))), dtype=float16, device=device)
        self.ForceBlueStored = torch.zeros((1, 2, math.floor(self.VelocityBlue.size(dim=1))), dtype=float16, device=device)
        self.ForceGreenStored = torch.zeros((1, 2, math.floor(self.VelocityGreen.size(dim=1))), dtype=float16, device=device)
        self.ForceYellowStored = torch.zeros((1, 2, math.floor(self.VelocityYellow.size(dim=1))), dtype=float16, device=device)
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
        self.calculateForces(self.PositionRedGpu, self.ForceRedStored, self.PositionRedGpu, self.RedRed)
        self.calculateForces(self.PositionRedGpu, self.ForceRedStored, self.PositionBlueGpu, self.RedBlue)
        self.calculateForces(self.PositionRedGpu, self.ForceRedStored, self.PositionGreenGpu, self.RedGreen)
        self.calculateForces(self.PositionRedGpu, self.ForceRedStored, self.PositionYellowGpu, self.RedYellow)
        self.calculateForces(self.PositionBlueGpu, self.ForceBlueStored, self.PositionRedGpu, self.BlueRed)
        self.calculateForces(self.PositionBlueGpu, self.ForceBlueStored, self.PositionBlueGpu, self.BlueBlue)
        self.calculateForces(self.PositionBlueGpu, self.ForceBlueStored, self.PositionGreenGpu, self.BlueGreen)
        self.calculateForces(self.PositionBlueGpu, self.ForceBlueStored, self.PositionYellowGpu, self.BlueYellow)
        self.calculateForces(self.PositionGreenGpu, self.ForceGreenStored, self.PositionRedGpu, self.GreenRed)
        self.calculateForces(self.PositionGreenGpu, self.ForceGreenStored, self.PositionBlueGpu, self.GreenBlue)
        self.calculateForces(self.PositionGreenGpu, self.ForceGreenStored, self.PositionGreenGpu, self.GreenGreen)
        self.calculateForces(self.PositionGreenGpu, self.ForceGreenStored, self.PositionYellowGpu, self.GreenYellow)
        self.calculateForces(self.PositionYellowGpu, self.ForceYellowStored, self.PositionRedGpu, self.YellowRed)
        self.calculateForces(self.PositionYellowGpu, self.ForceYellowStored, self.PositionBlueGpu, self.YellowBlue)
        self.calculateForces(self.PositionYellowGpu, self.ForceYellowStored, self.PositionGreenGpu, self.YellowGreen)
        self.calculateForces(self.PositionYellowGpu, self.ForceYellowStored, self.PositionYellowGpu, self.YellowYellow)

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
        clock.tick(240)
    print("average time: " + str(sum(timeList)/len(timeList)))
    pygame.quit()
        
if __name__ == "__main__":
    main()