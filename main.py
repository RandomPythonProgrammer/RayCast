import math

import PIL.ImageDraw
import numpy
import pyglet
from PIL import Image
from numba import njit
from numba.pycc import CC
cc = CC("game")

VIEW_RANGE = 1000
BRIGHTNESS = 175
SHOW_FPS = False

HEIGHT = 2
WIDTH = 0.01


@njit(cache=True)
@cc.export('get_lines', 'int64[:, :](int64, float64, int64, float64, float64, int64, int64, uint8[:, :, :])')
def get_lines(real_width: int, view_angle: float, height: int, scale: float, rotation: float, x: int, y: int, map_data: numpy.ndarray):
    half_width = real_width/2
    columns = range(-round(half_width), round(half_width))
    return_array = numpy.zeros(shape=(len(columns), 6), dtype="int64")
    rotation_offset = view_angle / real_width
    for column in columns:
        angle = math.radians(rotation_offset * column - rotation)
        result = cast_ray(angle, VIEW_RANGE, map_data, x, y)
        color = result[:3]
        distance, px, py = result[4:]
        distance *= (WIDTH/HEIGHT)
        if distance > 0:
            shade = 1 / numpy.power(max(distance, 1), 2.5)
            color = (color * shade).clip(0, BRIGHTNESS)
        r, g, b = color
        column_x = round((column + real_width / 2) * scale)
        column_height = round(height / distance)
        height_offset = round((height - column_height) / 2)
        return_array[round(column + half_width)] = numpy.array([column_x, height_offset, column_height, r, g, b], dtype="int64")
    return return_array


@njit(cache=True)
@cc.export('cast_ray', 'float64[:](float64, int64, uint8[:, :, :], int64, int64)')
def cast_ray(angle: float, limit: int, map_data: numpy.ndarray, x: int, y: int):
    width, height, depth = map_data.shape
    dx, dy = numpy.cos(angle), numpy.sin(angle)
    for i in range(limit):
        position = round(x + (dx * i)), round((height - 1) - y + (dy * i))
        px, py = position
        if 0 <= py < height and 0 <= px < width:
            temp_color = map_data[py][px]
            if temp_color[-1] > 0:
                distance = numpy.sqrt(numpy.square((px - x)) + numpy.square((py - (height - 1 - y))))
                r, g, b, a = temp_color
                return numpy.array([r, g, b, a, distance, px, py], dtype="float64")
        else:
            return numpy.array([0, 0, 0, 0, -1, 0, 0], dtype="float64")
    return numpy.array([0, 0, 0, 0, -1, 0, 0], dtype="float64")


class Game(pyglet.window.Window):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.set_vsync(False)
        self.scale = 8
        self.x = self.y = self.rotation = 0
        self.real_width = round(self.width / self.scale)
        self.view_angle = 90
        self.batch = pyglet.graphics.Batch()
        self.lines = []
        self.keys = pyglet.window.key.KeyStateHandler()

        self.map = Image.open("map2.png")
        self.map_data = numpy.asarray(self.map)

    def render(self, dt: float):
        self.cast_rays()
        self.clear()
        self.batch.draw()

    def cast_rays(self):
        self.lines.clear()
        lines = get_lines(self.real_width, self.view_angle, self.height, self.scale, self.rotation, self.x, self.y, self.map_data)
        for i in range(lines.shape[0]):
            x, height_offset, height, r, g, b = lines[i]
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
            r, g, b, a, distance, px, py = cast_ray(angle, VIEW_RANGE, self.map_data, self.x, self.y)
            if distance >= 0:
                draw.line(((self.x, image.height - 1 - self.y), (px, py)))
        image.show()

    def update(self, dt: float):
        speed = 150 * dt
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
        try:
            print(1/dt)
        except ZeroDivisionError:
            pass


if __name__ == '__main__':
    print("compiling...")
    cc.compile()
    print("done.")
    game = Game(1024, 640)
    pyglet.clock.schedule(game.render)
    pyglet.clock.schedule(game.update)
    pyglet.app.run()
