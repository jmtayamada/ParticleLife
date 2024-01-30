# importing the required libraries
import pygame
from pygame.locals import *
import random
import math
import numpy as np
import torch

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

# lifeFormList = np.array([])
lifeFormList = []

k_powerConstant = 1
k_maxCreatures = 7

device = torch.device("mps")


class Particle():

    def __init__(self, x: int, y: int, color: tuple) -> None:
        self.timeAlive = 0
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.color = color
        self.health = 100
        self.queuedDamage = 0
        if self.color == grey:
            self.health *= 10
        self.maxHealth = self.health
    
    def drawParticle(self):
        if self.color != grey:
            pygame.draw.rect(screen, self.color, (self.x - 1, self.y - 1 , 3, 3))
        else:
            pygame.draw.rect(screen, self.color, (self.x - 2, self.y - 2 , 5, 5))

    def getDistance(self, x, y):
        xDif = self.x - x
        yDif = self.y - y
        return math.sqrt(xDif*xDif + yDif*yDif)

    def applyForces(self):
        self.x += self.vx
        self.y += self.vy

        # now reversing the particles
        # when they hit the wall
        if self.x < 0:
            self.x = 0
            self.vx *= -1
        elif self.x > 700:
            self.x = 700
            self.vx *= -1
        if self.y < 0:
            self.y = 0
            self.vy *= -1
        elif self.y > 500:
            self.y = 500
            self.vy *= -1
            
    def onDeath(self):
        nearList = []
        for c in lifeFormList:
            for particle in c.particles:
                if self.getDistance(particle.x, particle.y) < 25:
                    nearList.append(particle)
        for particle in nearList:
            particle.health += math.ceil(self.timeAlive * self.maxHealth / len(nearList))


class Core():

    def __init__(self, x: int, y: int, size: int, ratio = tuple([0, 0, 0, 0])) -> None:
        self.x = x
        self.y = y
        self.rules = {
            "interior": {
                "yellow > yellow": random.randint(-100, 90)/100,
                "yellow > red": random.randint(-100, 90)/100,
                "yellow > green": random.randint(-100, 90)/100,
                "yellow > blue": random.randint(-100, 90)/100,
                "yellow > grey": random.randint(-100, 90)/100,
                "red > yellow": random.randint(-100, 90)/100,
                "red > red": random.randint(-100, 90)/100,
                "red > green": random.randint(-100, 90)/100,
                "red > blue": random.randint(-100, 90)/100,
                "red > grey": random.randint(-100, 90)/100,
                "green > yellow": random.randint(-100, 90)/100,
                "green > red": random.randint(-100, 90)/100,
                "green > green": random.randint(-100, 90)/100,
                "green > blue": random.randint(-100, 90)/100,
                "green > grey": random.randint(-100, 90)/100,
                "blue > yellow": random.randint(-100, 90)/100,
                "blue > red": random.randint(-100, 90)/100,
                "blue > green": random.randint(-100, 90)/100,
                "blue > blue": random.randint(-100, 90)/100,
                "blue > grey": random.randint(-100, 90)/100,
                "grey > yellow": random.randint(-100, 90)/100,
                "grey > red": random.randint(-100, 90)/100,
                "grey > green": random.randint(-100, 90)/100,
                "grey > blue": random.randint(-100, 90)/100,
            },
            "exterior": {
                "yellow > yellow": random.randint(-100, 100)/100,
                "yellow > red": random.randint(-100, 100)/100,
                "yellow > green": random.randint(-100, 100)/100,
                "yellow > blue": random.randint(-100, 100)/100,
                "yellow > grey": random.randint(-100, 100)/100,
                "red > yellow": random.randint(-100, 100)/100,
                "red > red": random.randint(-100, 100)/100,
                "red > green": random.randint(-100, 100)/100,
                "red > blue": random.randint(-100, 100)/100,
                "red > grey": random.randint(-100, 100)/100,
                "green > yellow": random.randint(-100, 100)/100,
                "green > red": random.randint(-100, 100)/100,
                "green > green": random.randint(-100, 100)/100,
                "green > blue": random.randint(-100, 100)/100,
                "green > grey": random.randint(-100, 100)/100,
                "blue > yellow": random.randint(-100, 100)/100,
                "blue > red": random.randint(-100, 100)/100,
                "blue > green": random.randint(-100, 100)/100,
                "blue > blue": random.randint(-100, 100)/100,
                "blue > grey": random.randint(-100, 100)/100,
                "grey > yellow": random.randint(-100, 100)/100,
                "grey > red": random.randint(-100, 100)/100,
                "grey > green": random.randint(-100, 100)/100,
                "grey > blue": random.randint(-100, 100)/100,
                "grey > grey": random.randint(-100, 100)/1000,
            }
        }
        
        self.size = size + 1

        self.ColorRatios = []
        for x in range(ratio[0]):
            self.ColorRatios.append(red)
        for x in range(ratio[1]):
            self.ColorRatios.append(blue)
        for x in range(ratio[2]):
            self.ColorRatios.append(green)
        for x in range(ratio[3]):
            self.ColorRatios.append(yellow)
        self.particles = []
        self.particles.append(Particle(self.x, self.y, grey))
        for x in range(size):
            self.particles.append(Particle(random.randint(self.x - 25, self.x + 25), random.randint(self.y - 25, self.y + 25), random.choice(self.ColorRatios)))

        self.coreParticle = None
        for particle in self.particles:
            if particle.color == grey:
                self.coreParticle = particle
                break

    def calculateForces(self, color: tuple, interior: bool, rule: float, particleList: list):
        for particle in particleList:
            fx = 0
            fy = 0

            if interior:
                for j in self.particles:
                    if j.color == color:
                        dx = particle.x - j.x
                        dy = particle.y - j.y
                        dis = math.sqrt(dx*dx + dy*dy)

                        if (dis > 0):
                            F = rule/dis
                            fx += (F*dx)
                            fy += (F*dy)

            if not interior:
                for c in lifeFormList:
                    if c != self:
                        for j in c.particles:
                            if j.color == color:
                                dx = particle.x - j.x
                                dy = particle.y - j.y
                                dis = math.sqrt(dx*dx + dy*dy)

                                if (dis > 0):
                                    F = rule/dis
                                    fx += (F*dx)
                                    fy += (F*dy)

            particle.vx = (particle.vx + fx)*0.5
            particle.vy = (particle.vy + fy)*0.5

    def ParticleCollisions(self):
        otherparticles = []
        for core in lifeFormList:
            if core != self:
                for particle in core.particles:
                    otherparticles.append(particle)
        for particle in self.particles:
            for other in otherparticles:
                if (particle.x == other.x and particle.y == other.y):
                    other.queuedDamage += particle.getDistance(self.x, self.y) * k_powerConstant

    def update(self):
        selfexists = False
        for p in self.particles:
            if p.color == grey:
                selfexists = True
                break
        if not selfexists:
            lifeFormList.remove(self)
            return

        redList = [x for x in self.particles if x.color == red]
        blueList = [x for x in self.particles if x.color == blue]
        greenList = [x for x in self.particles if x.color == green]
        yellowList = [x for x in self.particles if x.color == yellow]
        coreList = [x for x in self.particles if x.color == grey]

        self.calculateForces(red, True, self.rules["interior"]["red > red"], redList)
        self.calculateForces(blue, True, self.rules["interior"]["red > blue"], redList)
        self.calculateForces(green, True, self.rules["interior"]["red > green"], redList)
        self.calculateForces(yellow, True, self.rules["interior"]["red > yellow"], redList)
        self.calculateForces(grey, True, self.rules["interior"]["red > grey"], redList)

        self.calculateForces(red, True, self.rules["interior"]["blue > red"], blueList)
        self.calculateForces(blue, True, self.rules["interior"]["blue > blue"], blueList)
        self.calculateForces(green, True, self.rules["interior"]["blue > green"], blueList)
        self.calculateForces(yellow, True, self.rules["interior"]["blue > yellow"], blueList)
        self.calculateForces(grey, True, self.rules["interior"]["blue > grey"], blueList)

        self.calculateForces(red, True, self.rules["interior"]["green > red"], greenList)
        self.calculateForces(blue, True, self.rules["interior"]["green > blue"], greenList)
        self.calculateForces(green, True, self.rules["interior"]["green > green"], greenList)
        self.calculateForces(yellow, True, self.rules["interior"]["green > yellow"], greenList)
        self.calculateForces(grey, True, self.rules["interior"]["green > grey"], greenList)

        self.calculateForces(red, True, self.rules["interior"]["yellow > red"], yellowList)
        self.calculateForces(blue, True, self.rules["interior"]["yellow > blue"], yellowList)
        self.calculateForces(green, True, self.rules["interior"]["yellow > green"], yellowList)
        self.calculateForces(yellow, True, self.rules["interior"]["yellow > yellow"], yellowList)
        self.calculateForces(grey, True, self.rules["interior"]["yellow > grey"], yellowList)

        self.calculateForces(red, True, self.rules["interior"]["grey > red"], coreList)
        self.calculateForces(blue, True, self.rules["interior"]["grey > blue"], coreList)
        self.calculateForces(green, True, self.rules["interior"]["grey > green"], coreList)
        self.calculateForces(yellow, True, self.rules["interior"]["grey > yellow"], coreList)

        
        self.calculateForces(red, False, self.rules["exterior"]["red > red"], redList)
        self.calculateForces(blue, False, self.rules["exterior"]["red > blue"], redList)
        self.calculateForces(green, False, self.rules["exterior"]["red > green"], redList)
        self.calculateForces(yellow, False, self.rules["exterior"]["red > yellow"], redList)
        self.calculateForces(grey, False, self.rules["exterior"]["red > grey"], redList)

        self.calculateForces(red, False, self.rules["exterior"]["blue > red"], blueList)
        self.calculateForces(blue, False, self.rules["exterior"]["blue > blue"], blueList)
        self.calculateForces(green, False, self.rules["exterior"]["blue > green"], blueList)
        self.calculateForces(yellow, False, self.rules["exterior"]["blue > yellow"], blueList)
        self.calculateForces(grey, False, self.rules["exterior"]["blue > grey"], blueList)

        self.calculateForces(red, False, self.rules["exterior"]["green > red"], greenList)
        self.calculateForces(blue, False, self.rules["exterior"]["green > blue"], greenList)
        self.calculateForces(green, False, self.rules["exterior"]["green > green"], greenList)
        self.calculateForces(yellow, False, self.rules["exterior"]["green > yellow"], greenList)
        self.calculateForces(grey, False, self.rules["exterior"]["green > grey"], greenList)

        self.calculateForces(red, False, self.rules["exterior"]["yellow > red"], yellowList)
        self.calculateForces(blue, False, self.rules["exterior"]["yellow > blue"], yellowList)
        self.calculateForces(green, False, self.rules["exterior"]["yellow > green"], yellowList)
        self.calculateForces(yellow, False, self.rules["exterior"]["yellow > yellow"], yellowList)
        self.calculateForces(grey, False, self.rules["exterior"]["yellow > grey"], yellowList)

        self.calculateForces(red, False, self.rules["exterior"]["grey > red"], coreList)
        self.calculateForces(blue, False, self.rules["exterior"]["grey > blue"], coreList)
        self.calculateForces(green, False, self.rules["exterior"]["grey > green"], coreList)
        self.calculateForces(yellow, False, self.rules["exterior"]["grey > yellow"], coreList)
        self.calculateForces(grey, False, self.rules["exterior"]["grey > grey"], coreList)

        for obj in self.particles:
            obj.applyForces()
            obj.health -= obj.queuedDamage
            if obj.health <= 0:
                obj.onDeath()
                self.particles.remove(obj)
            if obj.health < obj.maxHealth:
                obj.health += 5
            obj.timeAlive += 1

        self.x = self.coreParticle.x
        self.y = self.coreParticle.y

        self.ParticleCollisions()

        for obj in self.particles:
            obj.drawParticle()

        for x in range(self.size - len(self.particles)):
            if random.randint(0, 1) == 1:
                self.particles.append(Particle(self.x, self.y, random.choice(self.ColorRatios)))
                

def main():
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill(black)
        for x in range(k_maxCreatures - len(lifeFormList)):
            lifeFormList.append(Core(random.randint(0, 700), random.randint(0, 500), 100, tuple([random.randint(0, 10), random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)])))
        for obj in lifeFormList:
            obj.update()
        pygame.display.update()
        clock.tick(120)
    pygame.quit()


if __name__ == "__main__":
    main()