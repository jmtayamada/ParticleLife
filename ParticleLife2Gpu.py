import pygame
from pygame.locals import *
import random
from pygame.sprite import _Group
import torch
from torch import tensor, float32, Tensor
import math
import time

k_maxCreatures = 1

red = (255, 0, 0)
blue = (0, 0, 255)
green = (0, 255, 0)
yellow = (255, 255, 0)
black = (0, 0, 0)
grey = (122, 122, 122)

pygame.init()
width = 700
height = 500
screen = pygame.display.set_mode((width + 1, height + 1))
pygame.display.set_caption("Particle Life")
screen.fill(black)

clock = pygame.time.Clock()

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
if torch.backends.mps.is_available():
    print("running on MPS")
elif torch.cuda.is_available():
    print("running on CUDA")
else:
    print("running on cpu")
    
lifeFormList = []

class Particle(pygame.sprite.Sprite):
    def __init__(self) -> None:
        super().__init__()

class ParticleLife():
    
    def __init__(self, size: int, colorRatios: []) -> None:        
        # creating color ratios for particles
        tot = colorRatios[0] + colorRatios[1] + colorRatios[2] + colorRatios[3]
        redRatio = colorRatios[0]/tot
        blueRatio = colorRatios[1]/tot
        greenRatio = colorRatios[2]/tot
        yellowRatio = colorRatios[3]/tot
        currentParticleDif = tot - (math.floor(redRatio * size) + math.floor(blueRatio * size) + math.floor(greenRatio * size) + math.floor(yellowRatio * size))
        
        # create tensors for x and y values for each color
        self.xTensorsRed = tensor([random.randint(0, width) for x in range(math.floor(redRatio * size))], dtype=int, device=device)
        self.xTensorsBlue = tensor([random.randint(0, width) for x in range(math.floor(blueRatio * size))], dtype=int, device=device)
        self.xTensorsGreen = tensor([random.randint(0, width) for x in range(math.floor(greenRatio * size))], dtype=int, device=device)
        self.xTensorsYellow = tensor([random.randint(0, width) for x in range(math.floor(yellowRatio * size))], dtype=int, device=device)
        self.yTensorsRed = tensor([random.randint(0, height) for x in range(math.floor(redRatio * size))], dtype=int, device=device)
        self.yTensorsBlue = tensor([random.randint(0, height) for x in range(math.floor(blueRatio * size))], dtype=int, device=device)
        self.yTensorsGreen = tensor([random.randint(0, height) for x in range(math.floor(greenRatio * size))], dtype=int, device=device)
        self.yTensorsYellow = tensor([random.randint(0, height) for x in range(math.floor(yellowRatio * size))], dtype=int, device=device)
        
        # make sure total particle num equals the total particles
        counter = 0
        for num in range(currentParticleDif):
            if counter == 0:
                self.xTensorsRed = torch.cat((self.xTensorsRed, tensor([random.randint(0, width)], dtype=int, device=device)))
                self.yTensorsRed = torch.cat((self.yTensorsRed, tensor([random.randint(0, height)], dtype=int, device=device)))
            elif counter == 1:
                self.xTensorsBlue = torch.cat((self.xTensorsBlue, tensor([random.randint(0, width)], dtype=int, device=device)))
                self.yTensorsBlue = torch.cat((self.yTensorsBlue, tensor([random.randint(0, height)], dtype=int, device=device)))
            else:
                self.xTensorsGreen = torch.cat((self.xTensorsGreen, tensor([random.randint(0, width)], dtype=int, device=device)))
                self.yTensorsGreen = torch.cat((self.yTensorsGreen, tensor([random.randint(0, height)], dtype=int, device=device)))
        
        # create vx and vy tensors for each color
        self.vxTensorsRed = torch.zeros([self.xTensorsRed.size(dim=0),], dtype=float32, device=device)
        self.vyTensorsRed = torch.zeros([self.yTensorsRed.size(dim=0),], dtype=float32, device=device)
        self.vxTensorsBlue = torch.zeros([self.xTensorsBlue.size(dim=0),], dtype=float32, device=device)
        self.vyTensorsBlue = torch.zeros([self.yTensorsBlue.size(dim=0),], dtype=float32, device=device)
        self.vxTensorsGreen = torch.zeros([self.xTensorsGreen.size(dim=0),], dtype=float32, device=device)
        self.vyTensorsGreen = torch.zeros([self.yTensorsGreen.size(dim=0),], dtype=float32, device=device)
        self.vxTensorsYellow = torch.zeros([self.xTensorsYellow.size(dim=0),], dtype=float32, device=device)
        self.vyTensorsYellow = torch.zeros([self.yTensorsYellow.size(dim=0),], dtype=float32, device=device)
        
        # rules
        self.RedRed = random.randint(-100, 100)/100
        self.RedBlue = random.randint(-100, 100)/100
        self.RedGreen = random.randint(-100, 100)/100
        self.RedYellow = random.randint(-100, 100)/100
        self.BlueRed = random.randint(-100, 100)/100
        self.BlueBlue = random.randint(-100, 100)/100
        self.BlueGreen = random.randint(-100, 100)/100
        self.BlueYellow = random.randint(-100, 100)/100
        self.GreenRed = random.randint(-100, 100)/100
        self.GreenBlue = random.randint(-100, 100)/100
        self.GreenGreen = random.randint(-100, 100)/100
        self.GreenYellow = random.randint(-100, 100)/100
        self.YellowRed = random.randint(-100, 100)/100
        self.YellowBlue = random.randint(-100, 100)/100
        self.YellowGreen = random.randint(-100, 100)/100
        self.YellowYellow = random.randint(-100, 100)/100
        
        # sprite groups for use in drawing
        self.redGroup = pygame.sprite.Group()
        self.blueGroup = pygame.sprite.Group()
        self.greenGroup = pygame.sprite.Group()
        self.yellowGroup = pygame.sprite.Group()
        
    def calculateForces(self, xTensor1: Tensor, yTensor1: Tensor, vxTensor1: Tensor, vyTensor1: Tensor, xTensor2: Tensor, yTensor2: Tensor, rule: int):
        # create new tensors for x and y tensors
        xTensors1 = xTensor1.repeat(xTensor2.size(dim=0),)
        yTensors1 = yTensor1.repeat(yTensor2.size(dim=0),)
        
        # create new tensors for x2 and y2 tensors
        xTensors2 = xTensor2.repeat_interleave(xTensor1.size(dim=0))
        yTensors2 = yTensor2.repeat_interleave(yTensor1.size(dim=0))
        
        # create tensors for fx and fy of particles
        fxTensor = torch.zeros((xTensors1.size(dim=0),), device=device)
        fyTensor = torch.zeros((yTensors1.size(dim=0),), device=device)
        
        # calculate distance between particles
        dx = xTensors1.sub(xTensors2)
        dy = yTensors1.sub(yTensors2)
        dis = (dx**2 + dy**2)**.5
        
        # calculate forces
        F = tensor(torch.reciprocal((dis + .0000000001)) * rule, dtype=float32, device=device)
        fxTensor += F*dx
        fyTensor += F*dy
                
        # reshape fx and fy tensors to match with xTensor1 and yTensor1, then add the different rows to get tensors with the same shape as x and y tensor
        fxTensor = fxTensor.reshape(xTensor1.size(dim=0), xTensor2.size(dim=0))
        fxTensor = torch.sum(fxTensor, 0)
        fyTensor = fyTensor.reshape(yTensor1.size(dim=0), yTensor2.size(dim=0))
        fyTensor = torch.sum(fyTensor, 0)
        
        # forces to velocity
        vxTensor1 = (vxTensor1 + fxTensor)*0.5
        vyTensor1 = (vyTensor1 + fyTensor)*0.5
        
        # velocity to postion
        xTensor1.add_(vxTensor1.long())
        yTensor1.add_(vyTensor1.long())
        
        # make sure particles stay on screen
        xTensor1.clamp_(0, 700)
        yTensor1.clamp_(0, 500)
        
        # reverse velocity if object is outside of the bounds
        vxTensor1.multiply_(torch.where(xTensor1 == width, -1.0, 1.0))
        vyTensor1.multiply_(torch.where(yTensor1 == height, -1.0, 1.0))
        vxTensor1.multiply_(torch.where(xTensor1 == 0, -1.0, 1.0))
        vyTensor1.multiply_(torch.where(yTensor1 == 0, -1.0, 1.0))
        
    def draw(self):
        for i in range(self.xTensorsRed.size(dim=0)):
            pygame.draw.rect(screen, red, (self.xTensorsRed[i] - 1, self.yTensorsRed[i] - 1, 3, 3))
        for i in range(self.xTensorsBlue.size(dim=0)):
            pygame.draw.rect(screen, blue, (self.xTensorsBlue[i] - 1, self.yTensorsBlue[i].item() - 1, 3, 3))
        for i in range(self.xTensorsGreen.size(dim=0)):
            pygame.draw.rect(screen, green, (self.xTensorsGreen[i] - 1, self.yTensorsGreen[i] - 1, 3, 3))
        for i in range(self.xTensorsYellow.size(dim=0)):
            pygame.draw.rect(screen, yellow, (self.xTensorsYellow[i] - 1, self.yTensorsYellow[i] - 1, 3, 3))
        
    def update(self):
        self.calculateForces(self.xTensorsRed, self.yTensorsRed, self.vxTensorsRed, self.vyTensorsRed, self.xTensorsRed, self.yTensorsRed, self.RedRed)
        self.calculateForces(self.xTensorsRed, self.yTensorsRed, self.vxTensorsRed, self.vyTensorsRed, self.xTensorsBlue, self.yTensorsBlue, self.RedBlue)
        self.calculateForces(self.xTensorsRed, self.yTensorsRed, self.vxTensorsRed, self.vyTensorsRed, self.xTensorsGreen, self.yTensorsGreen, self.RedGreen)
        self.calculateForces(self.xTensorsRed, self.yTensorsRed, self.vxTensorsRed, self.vyTensorsRed, self.xTensorsYellow, self.yTensorsYellow, self.RedYellow)
        self.calculateForces(self.xTensorsBlue, self.yTensorsBlue, self.vxTensorsBlue, self.vyTensorsBlue, self.xTensorsRed, self.yTensorsRed, self.BlueRed)
        self.calculateForces(self.xTensorsBlue, self.yTensorsBlue, self.vxTensorsBlue, self.vyTensorsBlue, self.xTensorsBlue, self.yTensorsBlue, self.BlueBlue)
        self.calculateForces(self.xTensorsBlue, self.yTensorsBlue, self.vxTensorsBlue, self.vyTensorsBlue, self.xTensorsGreen, self.yTensorsGreen, self.BlueGreen)
        self.calculateForces(self.xTensorsBlue, self.yTensorsBlue, self.vxTensorsBlue, self.vyTensorsBlue, self.xTensorsYellow, self.yTensorsYellow, self.BlueYellow)
        self.calculateForces(self.xTensorsGreen, self.yTensorsGreen, self.vxTensorsGreen, self.vyTensorsGreen, self.xTensorsRed, self.yTensorsRed, self.GreenRed)
        self.calculateForces(self.xTensorsGreen, self.yTensorsGreen, self.vxTensorsGreen, self.vyTensorsGreen, self.xTensorsBlue, self.yTensorsBlue, self.GreenBlue)
        self.calculateForces(self.xTensorsGreen, self.yTensorsGreen, self.vxTensorsGreen, self.vyTensorsGreen, self.xTensorsGreen, self.yTensorsGreen, self.GreenGreen)
        self.calculateForces(self.xTensorsGreen, self.yTensorsGreen, self.vxTensorsGreen, self.vyTensorsGreen, self.xTensorsYellow, self.yTensorsYellow, self.GreenYellow)
        self.calculateForces(self.xTensorsYellow, self.yTensorsYellow, self.vxTensorsYellow, self.vyTensorsYellow, self.xTensorsRed, self.yTensorsRed, self.YellowRed)
        self.calculateForces(self.xTensorsYellow, self.yTensorsYellow, self.vxTensorsYellow, self.vyTensorsYellow, self.xTensorsBlue, self.yTensorsBlue, self.YellowBlue)
        self.calculateForces(self.xTensorsYellow, self.yTensorsYellow, self.vxTensorsYellow, self.vyTensorsYellow, self.xTensorsGreen, self.yTensorsGreen, self.YellowGreen)
        self.calculateForces(self.xTensorsYellow, self.yTensorsYellow, self.vxTensorsYellow, self.vyTensorsYellow, self.xTensorsYellow, self.yTensorsYellow, self.YellowYellow)

        self.draw()
        


def main():
    running = True
    particleLife = ParticleLife(512, [1, 1, 1, 1])
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill(black)
        
        particleLife.update()

        pygame.display.update()
        clock.tick(120)
    pygame.quit()
        
if __name__ == "__main__":
    main()