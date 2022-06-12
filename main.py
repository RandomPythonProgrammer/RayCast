import math

import PIL.ImageDraw
import numpy
import pyglet
from PIL import Image
from numba import njit


@njit
def get_lines(real_width: int, view_angle: float, height: int, scale: float, rotation: float, x: int, y: int, map_data: numpy.ndarray):
    columns = range(-round(real_width / 2), round(real_width / 2))
    return_array = numpy.ndarray(shape=(len(columns), 6), dtype="int64")
    index = 0
    for column in columns:
        rotation_offset = view_angle / real_width
        angle = math.radians(rotation_offset * column - rotation)
        r, g, b, a, distance, px, py = cast_ray(angle, 50, map_data, x, y)
        column_x = round((column + real_width / 2) * scale)
        column_height = round(height / max(distance, 1))
        height_offset = round((height - column_height) / 2)
        return_array[index] = numpy.array([column_x, height_offset, column_height, r, g, b], dtype="int64")
        index += 1
    return return_array


@njit
def cast_ray(angle: float, limit: int, map_data: numpy.ndarray, x: int, y: int):
    width, height, depth = map_data.shape
    dx, dy = numpy.cos(angle), numpy.sin(angle)
    for i in range(limit):
        position = round(x + (dx * i)), round((height - 1) - y + (dy * i))
        px, py = position
        if 0 <= py < height and 0 <= px < width:
            temp_color = map_data[py][px]
            if temp_color[-1] > 0:
                distance = ((px - x) ** 2 + (py - (height - 1 - y)) ** 2) ** 0.5
                r, g, b, a = temp_color
                return numpy.array([r, g, b, a, distance, px, py], dtype="float64")
        else:
            return numpy.array([0, 0, 0, 0, -1, 0, 0], dtype="float64")
    return numpy.array([0, 0, 0, 0, -1, 0, 0], dtype="float64")


class Game(pyglet.window.Window):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.scale = 0.1
        self.x = self.y = self.rotation = 0
        self.real_width = round(self.width / self.scale)
        self.view_angle = 90
        self.batch = pyglet.graphics.Batch()
        self.lines = []
        self.keys = pyglet.window.key.KeyStateHandler()

        self.map = Image.open("map1.png")
        self.map_data = numpy.asarray(self.map)

    def render(self, dt: float):
        self.cast_rays()
        self.clear()
        self.batch.draw()

    def cast_rays(self):
        self.lines.clear()
        lines = get_lines(self.real_width, self.view_angle, self.height, self.scale, self.rotation, self.x, self.y, self.map_data)

        for line in lines:
            x, height_offset, height, r, g, b = line
            self.lines.append(pyglet.shapes.Line(
                x, height_offset,
                x, height + height_offset,
                self.scale,
                (r, g, b),
                batch=self.batch)
            )

    def on_key_press(self, symbol, modifiers):
        self.keys[symbol] = True
        if symbol == pyglet.window.key.Z:
            self.show_casts()
        if symbol == pyglet.window.key.C:
            print(self.x, self.y, self.rotation)

    def on_key_release(self, symbol, modifiers):
        self.keys[symbol] = False

    def show_casts(self):
        image = self.map.copy()
        draw = PIL.ImageDraw.Draw(image)
        rotation_offset = self.view_angle / self.real_width

        for column in range(-round(self.real_width / 2), round(self.real_width / 2)):
            angle = math.radians(rotation_offset * column - self.rotation)
            r, g, b, a, distance, px, py = cast_ray(angle, 50, self.map_data, self.x, self.y)
            if distance >= 0:
                draw.line(((self.x, image.height - 1 - self.y), (px, py)))
        image.show()

    def update(self, dt):
        speed = 15 * dt
        rotation_speed = 50 * dt
        rotation = self.rotation
        radians = math.radians(rotation)
        if self.keys[pyglet.window.key.LEFT]:
            rotation += rotation_speed
        elif self.keys[pyglet.window.key.RIGHT]:
            rotation -= rotation_speed
        elif self.keys[pyglet.window.key.UP]:
            dx = math.cos(radians) * speed
            dy = math.sin(radians) * speed
            self.x += dx
            self.y += dy
        elif self.keys[pyglet.window.key.DOWN]:
            dx = math.cos(radians) * -speed
            dy = math.sin(radians) * -speed
            self.x += dx
            self.y += dy

        self.rotation = rotation % 360


if __name__ == '__main__':
    game = Game(1024, 640)
    pyglet.clock.schedule(game.render)
    pyglet.clock.schedule(game.update)
    pyglet.app.run()
