import pygame
from pygame.locals import *
import random
import torch
from torch import tensor, float32
import time

k_maxCreatures = 1

pygame.init()
width = 700
height = 500
color = "black"
screen = pygame.display.set_mode((width + 1, height + 1))
pygame.display.set_caption("Particle Life")
screen.fill(color)

clock = pygame.time.Clock()

red = (255, 0, 0)
blue = (0, 0, 255)
green = (0, 255, 0)
yellow = (255, 255, 0)
black = (0, 0, 0)
grey = (122, 122, 122)

if torch.backends.mps.is_available():
    print("running on MPS")
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

lifeFormList = []

class Particle():
    
    def __init__(self, x: tensor, y: tensor, color) -> None:
        self.x = tensor([x], dtype=int, device=device)
        self.y = tensor([y], dtype=int, device=device)
        self.vx = tensor([0], dtype=int, device=device)
        self.vy = tensor([0], dtype=int, device=device)
        self.color = color
        
    def drawParticle(self):
        if self.color != grey:
            pygame.draw.rect(screen, self.color, (self.x.item() - 1, self.y.item() - 1, 3, 3))
        else:
            pygame.draw.rect(screen, self.color, (self.x.item() - 2, self.y.item() - 2, 5, 5))    

class Core():
    
    def __init__(self, x: int, y: int, size: int, colorRatio: list[int]) -> None:
        self.x = tensor([x], dtype=int, device=device)
        self.y = tensor([y], dtype=int, device=device)
        self.size = size + 1
        
        self.colorRatio = []
        for obj in range(colorRatio[0]):
            self.colorRatio.append(red)
        for obj in range(colorRatio[1]):
            self.colorRatio.append(blue)
        for obj in range(colorRatio[2]):
            self.colorRatio.append(green)
        for obj in range(colorRatio[3]):
            self.colorRatio.append(yellow)
        
        self.particles = [Particle(self.x + random.randint(-5, 5), self.y + random.randint(-5, 5), random.choice(self.colorRatio)) for x in range(size)]
        self.redParticles = [particle for particle in self.particles if particle.color == red]
        self.blueParticles = [particle for particle in self.particles if particle.color == blue]
        self.greenParticles = [particle for particle in self.particles if particle.color == green]
        self.yellowParticles = [particle for particle in self.particles if particle.color == yellow]
        self.core = Particle(self.x, self.y, grey)
        self.particles.append(self.core)
        
    def calculateForces(self, particleset1: list[Particle], particleset2: list[Particle], rule: float):
        # create tensors containing x and y values for the first set of particles
        xTensor = tensor([particle.x for particle in particleset1], dtype=float32, device=device)
        yTensor = tensor([particle.y for particle in particleset1], dtype=float32, device=device)
        
        # create tensors containg vx and vy values for the first set of particles
        vxTensor = tensor([particle.vx for particle in particleset1], dtype=float32, device=device)
        vyTensor = tensor([particle.vy for particle in particleset1], dtype=float32, device=device)
        
        # create tensors that will store total fx and fy values for the first set of particles
        fxTensor = tensor([0 for particle in particleset1], dtype=float32, device=device)
        fyTensor = tensor([0 for particle in particleset1], dtype=float32, device=device)
        
        # create tensors containing x and y values for the second set of particles
        xTensor2 = tensor([particle.x for particle in particleset2], dtype=float32, device=device)
        yTensor2 = tensor([particle.y for particle in particleset2], dtype=float32, device=device)
        
        # iterate over this loop for every value in the second set of particles
        # TODO: see if it's possible to do these calculations at once
        for i in range(len(particleset2)):
            # calculate distance between particles
            dx = xTensor - torch.full(size=(len(particleset1),), fill_value=xTensor2[i], dtype=float32, device=device)
            dy = yTensor - torch.full(size=(len(particleset1),), fill_value=yTensor2[i], dtype=float32, device=device)
            dis = (dx**2 + dy**2)**.5
            
            # calculate forces
            F = tensor(torch.reciprocal((dis + .0000000001)/rule), dtype=float32,device=device)
            fxTensor += F*dx
            fyTensor += F*dy
        
        # apply forces to vx and vy tensors
        vxTensor = (vxTensor + fxTensor)*0.5
        vyTensor = (vyTensor + fyTensor)*0.5
        
        # apply velocity tensors to position tensors
        xTensor.add_(vxTensor)
        yTensor.add_(vyTensor)
        
        # # make sure position values aren't greater than max limits
        # xTensor.ceil_(700)
        # xTensor.floor_(0)
        # yTensor.ceil_(500)
        # yTensor.floor_(0)
        
        # multiply velocity by -1 if particle is touching the edge
        vxTensor.multiply_(torch.where(xTensor == 700, -1.0, 1.0))
        vyTensor.multiply_(torch.where(yTensor == 500, -1.0, 1.0))
        vxTensor.multiply_(torch.where(xTensor == 0, -1.0, 1.0))
        vyTensor.multiply_(torch.where(yTensor == 0, -1.0, 1.0))
                
        # apply x, y, vx, and vy tensors to their respective particles
        for val in range(len(particleset1)):
            particleset1[val].vx = vxTensor[val]
            particleset1[val].vy = vyTensor[val]
            particleset1[val].x = vxTensor[val].round()
            particleset1[val].y = vyTensor[val].round()
            
    def update(self):
        self.calculateForces(self.redParticles, self.redParticles, random.randint(-100, 100)/100)
        self.calculateForces(self.redParticles, self.blueParticles, random.randint(-100, 100)/100)
        self.calculateForces(self.redParticles, self.greenParticles, random.randint(-100, 100)/100)
        self.calculateForces(self.redParticles, self.yellowParticles, random.randint(-100, 100)/100)
        self.calculateForces(self.redParticles, [self.core], random.randint(-100, 100)/100)
        self.calculateForces(self.blueParticles, self.redParticles, random.randint(-100, 100)/100)
        self.calculateForces(self.blueParticles, self.blueParticles, random.randint(-100, 100)/100)
        self.calculateForces(self.blueParticles, self.greenParticles, random.randint(-100, 100)/100)
        self.calculateForces(self.blueParticles, self.yellowParticles, random.randint(-100, 100)/100)
        self.calculateForces(self.blueParticles, [self.core], random.randint(-100, 100)/100)
        self.calculateForces(self.greenParticles, self.redParticles, random.randint(-100, 100)/100)
        self.calculateForces(self.greenParticles, self.blueParticles, random.randint(-100, 100)/100)
        self.calculateForces(self.greenParticles, self.greenParticles, random.randint(-100, 100)/100)
        self.calculateForces(self.greenParticles, self.yellowParticles, random.randint(-100, 100)/100)
        self.calculateForces(self.greenParticles, [self.core], random.randint(-100, 100)/100)
        self.calculateForces(self.yellowParticles, self.redParticles, random.randint(-100, 100)/100)
        self.calculateForces(self.yellowParticles, self.blueParticles, random.randint(-100, 100)/100)
        self.calculateForces(self.yellowParticles, self.greenParticles, random.randint(-100, 100)/100)
        self.calculateForces(self.yellowParticles, self.yellowParticles, random.randint(-100, 100)/100)
        self.calculateForces(self.yellowParticles, [self.core], random.randint(-100, 100)/100)
        
        for particle in self.particles:
            particle.drawParticle()
        
        
def main():
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill(black)
        
        for x in range(k_maxCreatures - len(lifeFormList)):
            lifeFormList.append(Core(0, 0, 100, [1, 1, 1, 1]))
        
        for obj in lifeFormList:
            obj.update()

        pygame.display.update()
        clock.tick()
    pygame.quit()
        
if __name__ == "__main__":
    main()