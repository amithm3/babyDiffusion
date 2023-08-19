import os
import random as rnd

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import PILToTensor


class ShapesGenerator:
    DIR = "./shapesdata/"

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def __init__(self, x: int, y: int):
        self._x = x
        self._y = y
        self._PTT = PILToTensor()
        if not os.path.exists(self.DIR): os.mkdir(self.DIR)

    def gen_shape(self, fill: str, points: list[tuple[float, float]]):
        with Image.new('RGB', (self._x, self._y), (255, 255, 255)) as im:
            draw = ImageDraw.Draw(im)
            draw.polygon(points, fill=fill)
            return im

    def gen_circle(self, fill: str, xy: tuple[float, float], r: float):
        with Image.new('RGB', (self._x, self._y), (255, 255, 255)) as im:
            draw = ImageDraw.Draw(im)
            draw.ellipse((xy[0] - r, xy[1] - r, xy[0] + r, xy[1] + r), fill=fill)
            return im

    def gen_regular(self, fill: str, xy: tuple[float, float], r: float, n: int):
        s = float(torch.rand(1) * 360)
        a = torch.deg2rad(torch.linspace(s - 360, s, n + 1))
        return self.gen_shape(fill, [(xy[0] + r * torch.cos(a), xy[1] + r * torch.sin(a)) for a in a])

    def data(self, mb: int = 1) -> tuple["torch.Tensor", list[str]]:
        batch = []
        names = []
        i = 0
        img_list = os.listdir(self.DIR)
        for file in rnd.sample(img_list, k=len(img_list)):
            if i == mb:
                yield torch.stack(batch), names
                batch = []
                names = []
                i = 0
            batch.append(self._PTT(Image.open(self.DIR + file)).float() / 255)
            names.append(file)
            i += 1


def gen_shapes_data(sg: "ShapesGenerator", shapes: list[str], colors: list[str], sizes: list[str]):
    ns = 0
    while ns < 36:
        for shape in shapes:
            col = colors[torch.randint(0, len(colors), (1,)).item()]
            r = torch.randint(min(sg.x, sg.y) // 8, max_ := min(sg.x, sg.y) // 2, (1,)).item()
            xy = int(torch.randint(r, sg.x - r, (1,))), int(torch.randint(r, sg.y - r, (1,)))

            match shape:
                case "circle":
                    im = sg.gen_circle(col, xy, r)
                case "triangle":
                    im = sg.gen_regular(col, xy, r, 3)
                case "square":
                    im = sg.gen_regular(col, xy, r, 4)
                case "pentagon":
                    im = sg.gen_regular(col, xy, r, 5)
                case _:
                    raise ValueError(f"Unknown shape: {shape}")

            rng = *xy, r

            pos = ""
            xx = (xy[0] - r) / (sg.x - 2 * r)
            if xx < 2 / 5:
                pos += "left"
            elif xx > 3 / 5:
                pos += "right"
            else:
                pos += "center"
            yy = (xy[1] - r) / (sg.y - 2 * r)
            if yy < 2 / 5:
                pos += "%top"
            elif yy > 3 / 5:
                pos += "%bottom"
            else:
                pos += "%middle"

            r /= max_
            if r < 1 / 5:
                size = sizes[0]
            elif r < 2 / 5:
                size = sizes[1]
            elif r < 3 / 5:
                size = sizes[2]
            elif r < 4 / 5:
                size = sizes[3]
            else:
                size = sizes[4]

            im.save(f"{sg.DIR}/{shape}-{size}-{col}-{pos}-{rng}.png")
            ns += 1
