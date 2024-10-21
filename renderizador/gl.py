#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <SEU NOME AQUI>
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

import time  # Para operações com tempo
import gpu  # Simula os recursos de uma GPU
import math  # Funções matemáticas
import numpy as np  # Biblioteca do Numpy


class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800  # largura da tela
    height = 600  # altura da tela
    near = 0.01  # plano de corte próximo
    far = 1000  # plano de corte distante
    perspective_matrix = np.identity(4)
    view_matrix = np.identity(4)
    transformation_matrix = np.identity(4)
    transform_stack = [np.identity(4)]

    z_buffer = np.full((width, height), np.inf)
    framebuffer = np.zeros((width, height, 3), dtype=np.uint8)

    current_texture = None
    current_tex_coords = []

    @staticmethod
    def calculate_mipmap_level(v0, v1, v2, t0, t1, t2, texture):
        def triangle_area(edge1, edge2):
            return 0.5 * np.abs(edge1[0] * edge2[1] - edge1[1] * edge2[0])

        edge1 = np.subtract(v1[:2], v0[:2])
        edge2 = np.subtract(v2[:2], v0[:2])

        tex_edge1 = np.subtract(t1[:2], t0[:2])
        tex_edge2 = np.subtract(t2[:2], t0[:2])

        screen_area = triangle_area(edge1, edge2)
        tex_area = triangle_area(tex_edge1, tex_edge2)

        if tex_area <= 0:
            return 0

        ratio = screen_area / tex_area
        level = max(0, min(int(np.log2(ratio)), len(texture) - 1))

        return level

    @staticmethod
    def get_texture_color(texture, u, v, level):
        h, w, _ = texture.shape

        u = u % 1.0
        v = v % 1.0

        x = int(u * (w - 1))
        y = int(v * (h - 1))

        color = texture[y, x] / 255.0
        return color

    @staticmethod
    def draw_pixel(coord, depth, color, transparency):
        x, y = coord

        if 0 <= x < GL.width and 0 <= y < GL.height:
            if depth < GL.z_buffer[x, y]:
                GL.z_buffer[x, y] = depth
                existing_color = GL.framebuffer[x, y] / 255.0
                new_color = np.array(color) / 255.0
                final_color = (
                    1 - transparency
                ) * new_color + transparency * existing_color
                final_color = np.clip(final_color, 0.0, 1.0)
                GL.framebuffer[x, y] = (final_color * 255).astype(np.uint8)
                gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, GL.framebuffer[x, y])

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polypoint2D
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).

        # # Exemplo:
        # pos_x = GL.width//2
        # pos_y = GL.height//2
        # gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, [255, 0, 0])  # altera pixel (u, v, tipo, r, g, b)
        # # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)

        try:
            emissive_color = colors["emissiveColor"]
        except KeyError:
            emissive_color = [1.0, 1.0, 1.0]
        r, g, b = [int(c * 255) for c in emissive_color]

        for i in range(0, len(point), 2):
            x = int(point[i])
            y = int(point[i + 1])
            gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, [r, g, b])

    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polyline2D
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        def draw_line(x1, y1, x2, y2, colors):
            """Função auxiliar para desenhar uma linha."""
            dist_x = abs(x2 - x1)
            dist_y = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            delta = dist_x - dist_y

            while True:
                GL.polypoint2D([x1, y1], colors)
                if x1 == x2 and y1 == y2:
                    break
                delta_2 = 2 * delta
                if delta_2 > -dist_y:
                    delta -= dist_y
                    x1 += sx
                if delta_2 < dist_x:
                    delta += dist_x
                    y1 += sy

        for i in range(0, len(lineSegments), 4):
            x1 = int(lineSegments[i])
            y1 = int(lineSegments[i + 1])
            x2 = int(lineSegments[i + 2])
            y2 = int(lineSegments[i + 3])
            draw_line(x1, y1, x2, y2, colors)

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Circle2D
        # Nessa função você receberá um valor de raio e deverá desenhar o contorno de
        # um círculo.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Circle2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).

        print("Circle2D : radius = {0}".format(radius))  # imprime no terminal
        print("Circle2D : colors = {0}".format(colors))  # imprime no terminal as cores

        # Exemplo:
        pos_x = GL.width // 2
        pos_y = GL.height // 2
        gpu.GPU.draw_pixel(
            [pos_x, pos_y], gpu.GPU.RGB8, [255, 0, 255]
        )  # altera pixel (u, v, tipo, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)

    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#TriangleSet2D
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).
        print("TriangleSet2D : vertices = {0}".format(vertices))  # imprime no terminal
        print(
            "TriangleSet2D : colors = {0}".format(colors)
        )  # imprime no terminal as cores
        ss_factor = 2
        ss_width = GL.width * ss_factor
        ss_height = GL.height * ss_factor
        ss_framebuffer = np.zeros((ss_height, ss_width, 3), dtype=np.float32)

        def edge_intersect_y(y, p0, p1):
            if p0[1] == p1[1]:
                return None
            if p0[1] > p1[1]:
                p0, p1 = p1, p0
            if not (p0[1] <= y <= p1[1]):
                return None
            t = (y - p0[1]) / (p1[1] - p0[1])
            return p0[0] + t * (p1[0] - p0[0])

        for i in range(0, len(vertices), 6):
            v0 = [coord * ss_factor for coord in vertices[i : i + 2]]
            v1 = [coord * ss_factor for coord in vertices[i + 2 : i + 4]]
            v2 = [coord * ss_factor for coord in vertices[i + 4 : i + 6]]

            min_y = int(min(v0[1], v1[1], v2[1]))
            max_y = int(max(v0[1], v1[1], v2[1]))

            for y in range(min_y, max_y + 1):
                intersections = []
                for p0, p1 in [(v0, v1), (v1, v2), (v2, v0)]:
                    x_intersect = edge_intersect_y(y, p0, p1)
                    if x_intersect is not None:
                        intersections.append(x_intersect)

                if len(intersections) >= 2:
                    intersections.sort()
                    x_start = int(intersections[0])
                    x_end = int(intersections[1])

                    for x in range(x_start, x_end + 1):
                        if 0 <= x < ss_width and 0 <= y < ss_height:
                            ss_framebuffer[y, x] = colors["emissiveColor"]

        for y in range(GL.height):
            for x in range(GL.width):
                color_sum = np.zeros(3)
                for dy in range(ss_factor):
                    for dx in range(ss_factor):
                        ss_x = min(x * ss_factor + dx, ss_width - 1)
                        ss_y = min(y * ss_factor + dy, ss_height - 1)
                        color_sum += ss_framebuffer[ss_y, ss_x]

                avg_color = (color_sum / (ss_factor * ss_factor)) * 255
                gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, avg_color.astype(int).tolist())

    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleSet
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.

        w, h = GL.width - 1, GL.height - 1
        screen_matrix = np.array(
            [[w / 2, 0, 0, w / 2], [0, -h / 2, 0, h / 2], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        perVertex = False
        if type(colors) == list and len(colors) != 0:
            perVertex = True

        triangleSet3D_input = []
        tex_coords = GL.current_tex_coords

        while point:
            X, Y, Z = point.pop(0), point.pop(0), point.pop(0)
            p = np.array([X, Y, Z, 1.0])

            p_transformed = GL.perspective_matrix @ GL.transform_stack[-1] @ p

            triangleSet3D_input.extend([p_transformed[0]])
            triangleSet3D_input.extend([p_transformed[1]])
            triangleSet3D_input.extend([p_transformed[2]])
            triangleSet3D_input.extend([p_transformed[3]])

        def barycentric_coords(x, y, v0, v1, v2):
            denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (
                v0[1] - v2[1]
            )
            w1 = ((v1[1] - v2[1]) * (x - v2[0]) + (v2[0] - v1[0]) * (y - v2[1])) / denom
            w2 = ((v2[1] - v0[1]) * (x - v2[0]) + (v0[0] - v2[0]) * (y - v2[1])) / denom
            w3 = 1 - w1 - w2
            return w1, w2, w3

        for i in range(0, len(triangleSet3D_input), 12):
            v0 = triangleSet3D_input[i : i + 4]
            v1 = triangleSet3D_input[i + 4 : i + 8]
            v2 = triangleSet3D_input[i + 8 : i + 12]

            v0_proj = v0[:3] / v0[3]
            v1_proj = v1[:3] / v1[3]
            v2_proj = v2[:3] / v2[3]

            v0_proj = screen_matrix @ np.append(v0_proj, 1)
            v1_proj = screen_matrix @ np.append(v1_proj, 1)
            v2_proj = screen_matrix @ np.append(v2_proj, 1)

            if type(GL.current_texture) != None and len(GL.current_tex_coords) != 0:
                t0 = tex_coords[(i // 12) * 6 : (i // 12) * 6 + 2]
                t1 = tex_coords[(i // 12) * 6 + 2 : (i // 12) * 6 + 4]
                t2 = tex_coords[(i // 12) * 6 + 4 : (i // 12) * 6 + 6]

            min_x = int(min(v0_proj[0], v1_proj[0], v2_proj[0]))
            max_x = int(max(v0_proj[0], v1_proj[0], v2_proj[0]))
            min_y = int(min(v0_proj[1], v1_proj[1], v2_proj[1]))
            max_y = int(max(v0_proj[1], v1_proj[1], v2_proj[1]))

            if perVertex:
                v0_r = colors.pop(0)
                v0_g = colors.pop(0)
                v0_b = colors.pop(0)
                v1_r = colors.pop(0)
                v1_g = colors.pop(0)
                v1_b = colors.pop(0)
                v2_r = colors.pop(0)
                v2_g = colors.pop(0)
                v2_b = colors.pop(0)
                transp = 0.0

            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    w1, w2, w3 = barycentric_coords(x, y, v0_proj, v1_proj, v2_proj)

                    if w1 >= 0 and w2 >= 0 and w3 >= 0:
                        z_inv = (w1 / v0[2]) + (w2 / v1[2]) + (w3 / v2[2])
                        z = 1 / z_inv

                        if perVertex:
                            color_r = (
                                w1 * v0_r / v0[2]
                                + w2 * v1_r / v1[2]
                                + w3 * v2_r / v2[2]
                            ) * z
                            color_g = (
                                w1 * v0_g / v0[2]
                                + w2 * v1_g / v1[2]
                                + w3 * v2_g / v2[2]
                            ) * z
                            color_b = (
                                w1 * v0_b / v0[2]
                                + w2 * v1_b / v1[2]
                                + w3 * v2_b / v2[2]
                            ) * z
                            color = [color_r, color_g, color_b]
                        elif (
                            type(GL.current_texture) != None
                            and len(GL.current_tex_coords) != 0
                        ):
                            v = (
                                w1 * t0[0] / v0[2]
                                + w2 * t1[0] / v1[2]
                                + w3 * t2[0] / v2[2]
                            ) * z
                            u = (
                                -(
                                    w1 * t0[1] / v0[2]
                                    + w2 * t1[1] / v1[2]
                                    + w3 * t2[1] / v2[2]
                                )
                                * z
                            )
                            level = GL.calculate_mipmap_level(
                                v0, v1, v2, t0, t1, t2, GL.current_texture
                            )
                            r, g, b, _ = GL.get_texture_color(
                                GL.current_texture, u, v, level
                            )
                            color = [r, g, b]
                            transp = 0.0
                        else:
                            transp = colors["transparency"]
                            color = colors["emissiveColor"]

                        GL.draw_pixel([x, y], z, [int(c * 255) for c in color], transp)

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Sets up the view and projection matrices."""

        # Extracting components for readability
        x, y, z = orientation[:3]
        t = orientation[3]

        # Precompute sin and cos for efficiency
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        one_minus_cos_t = 1 - cos_t

        # Rotation matrix for arbitrary axis (orientation vector)
        rotation_m = np.array(
            [
                [
                    cos_t + x * x * one_minus_cos_t,
                    x * y * one_minus_cos_t - z * sin_t,
                    x * z * one_minus_cos_t + y * sin_t,
                    0,
                ],
                [
                    y * x * one_minus_cos_t + z * sin_t,
                    cos_t + y * y * one_minus_cos_t,
                    y * z * one_minus_cos_t - x * sin_t,
                    0,
                ],
                [
                    z * x * one_minus_cos_t - y * sin_t,
                    z * y * one_minus_cos_t + x * sin_t,
                    cos_t + z * z * one_minus_cos_t,
                    0,
                ],
                [0, 0, 0, 1],
            ]
        )

        # Translation matrix (camera position)
        translate_m = np.array(
            [
                [1.0, 0.0, 0.0, -position[0]],
                [0.0, 1.0, 0.0, -position[1]],
                [0.0, 0.0, 1.0, -position[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # LookAt matrix: Rotate first, then translate
        look_at_matrix = rotation_m @ translate_m

        # Perspective projection matrix
        aspect_ratio = GL.width / GL.height
        near = GL.near
        far = GL.far
        top = near * np.tan(fieldOfView / 2)
        right = top * aspect_ratio

        perspective_m = np.array(
            [
                [near / right, 0.0, 0.0, 0.0],
                [0.0, near / top, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    -(far + near) / (far - near),
                    -2.0 * far * near / (far - near),
                ],
                [0.0, 0.0, -1.0, 0.0],
            ]
        )

        # Final matrix that combines LookAt and Perspective
        GL.perspective_matrix = perspective_m @ look_at_matrix

        # Optionally print matrices for debugging
        print("LookAt Matrix:\n", look_at_matrix)
        print("Perspective Matrix:\n", perspective_m)
        print("Combined Matrix:\n", GL.perspective_matrix)

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo para depois potencialmente usar em outras chamadas.
        # Quando começar a usar Transforms dentre de outros Transforms, mais a frente no curso
        # Você precisará usar alguma estrutura de dados pilha para organizar as matrizes.

        scale_m = np.array(
            [
                [scale[0], 0, 0, 0],
                [0, scale[1], 0, 0],
                [0, 0, scale[2], 0],
                [0, 0, 0, 1],
            ]
        )

        # Matriz de translação
        translation_m = np.array(
            [
                [1, 0, 0, translation[0]],
                [0, 1, 0, translation[1]],
                [0, 0, 1, translation[2]],
                [0, 0, 0, 1],
            ]
        )

        # Matriz de rotação
        x, y, z, t = rotation
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        one_minus_cos_t = 1 - cos_t
        rotation_m = np.array(
            [
                [
                    cos_t + x * x * one_minus_cos_t,
                    x * y * one_minus_cos_t - z * sin_t,
                    x * z * one_minus_cos_t + y * sin_t,
                    0,
                ],
                [
                    y * x * one_minus_cos_t + z * sin_t,
                    cos_t + y * y * one_minus_cos_t,
                    y * z * one_minus_cos_t - x * sin_t,
                    0,
                ],
                [
                    z * x * one_minus_cos_t - y * sin_t,
                    z * y * one_minus_cos_t + x * sin_t,
                    cos_t + z * z * one_minus_cos_t,
                    0,
                ],
                [0, 0, 0, 1],
            ]
        )

        # Combina as transformações
        transformation_m = translation_m @ rotation_m @ scale_m

        # Empilha a nova transformação
        GL.transform_stack.append(GL.transform_stack[-1] @ transformation_m)

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.

        if len(GL.transform_stack) > 1:
            GL.transform_stack.pop()

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleStripSet
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        color = np.array(colors.get("emissiveColor", [1.0, 1.0, 1.0])) * 255

        # Iterar sobre cada strip
        idx = 0
        for count in stripCount:
            for i in range(2, count):
                p1 = point[idx : idx + 3]
                p2 = point[idx + 3 : idx + 6]
                p3 = point[idx + 6 : idx + 9]

                GL.triangleSet(p1 + p2 + p3, colors)
                idx += 3

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        color = np.array(colors.get("emissiveColor", [1.0, 1.0, 1.0])) * 255

        idx = 0
        strip = []
        while idx < len(index):
            if index[idx] == -1:
                if len(strip) >= 3:
                    for i in range(2, len(strip)):
                        p1 = point[strip[i - 2] * 3 : (strip[i - 2] + 1) * 3]
                        p2 = point[strip[i - 1] * 3 : (strip[i - 1] + 1) * 3]
                        p3 = point[strip[i] * 3 : (strip[i] + 1) * 3]
                        GL.triangleSet(p1 + p2 + p3, colors)
                strip = []
            else:
                strip.append(index[idx])
            idx += 1

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size))  # imprime no terminal pontos
        print("Box : colors = {0}".format(colors))  # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def indexedFaceSet(
        coord,
        coordIndex,
        colorPerVertex,
        color,
        colorIndex,
        texCoord,
        texCoordIndex,
        colors,
        current_texture,
    ):
        """Função usada para renderizar IndexedFaceSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#IndexedFaceSet
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão não possui uma ordem oficial, mas em geral se o primeiro ponto com os dois
        # seguintes e depois este mesmo primeiro ponto com o terçeiro e quarto ponto. Por exemplo: numa
        # sequencia 0, 1, 2, 3, 4, -1 o primeiro triângulo será com os vértices 0, 1 e 2, depois serão
        # os vértices 0, 2 e 3, e depois 0, 3 e 4, e assim por diante, até chegar no final da lista.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.

        if current_texture:
            GL.current_texture = gpu.GPU.load_texture(current_texture[0])

        triangles = []
        triangle_colors = []
        new_points = [
            [coord[i], coord[i + 1], coord[i + 2]] for i in range(0, len(coord), 3)
        ]

        def add_texture_coordinates(i):
            t1 = texCoordIndex[i]
            t2 = texCoordIndex[i + 1]
            t3 = texCoordIndex[i + 2]
            GL.current_tex_coords.extend(
                [
                    texCoord[t1 * 2],
                    texCoord[t1 * 2 + 1],
                    texCoord[t2 * 2],
                    texCoord[t2 * 2 + 1],
                    texCoord[t3 * 2],
                    texCoord[t3 * 2 + 1],
                ]
            )

        def add_vertex_colors(i):
            c1 = color[colorIndex[i] * 3 : colorIndex[i] * 3 + 3]
            c2 = color[colorIndex[i + 1] * 3 : colorIndex[i + 1] * 3 + 3]
            c3 = color[colorIndex[i + 2] * 3 : colorIndex[i + 2] * 3 + 3]
            triangle_colors.extend(c1 + c2 + c3)

        i = 0
        while i < len(coordIndex) - 2:
            if coordIndex[i] == -1:
                i += 1
                continue

            if coordIndex[i + 1] == -1 or coordIndex[i + 2] == -1:
                i += 1
                continue

            v1, v2, v3 = coordIndex[i], coordIndex[i + 1], coordIndex[i + 2]

            triangles.extend(new_points[v1] + new_points[v2] + new_points[v3])

            if colorPerVertex and color and colorIndex:
                add_vertex_colors(i)
            elif texCoord and texCoordIndex:
                add_texture_coordinates(i)
            else:
                emissiveColor = colors["emissiveColor"]
                triangle_colors.extend(emissiveColor * 3)

            i += 1

        GL.triangleSet(triangles, triangle_colors)

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # Vértices da caixa em torno da origem (0, 0, 0)
        x, y, z = size[0] / 2, size[1] / 2, size[2] / 2
        vertices = [
            [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],  # Face de trás
            [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z],  # Face da frente
        ]
        
        # Triângulos para a face de trás
        triangles = [
            [0, 1, 2], [0, 2, 3],  # Face de trás
            [4, 5, 6], [4, 6, 7],  # Face da frente
            [0, 1, 5], [0, 5, 4],  # Lado inferior
            [2, 3, 7], [2, 7, 6],  # Lado superior
            [0, 3, 7], [0, 7, 4],  # Lado esquerdo
            [1, 2, 6], [1, 6, 5],  # Lado direito
        ]
        
        # Desenhar os triângulos com base nas coordenadas e aplicar a cor
        for tri in triangles:
            v1, v2, v3 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            gpu.GPU.draw_triangle(v1, v2, v3, colors)


    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        latitudes = 20
        longitudes = 20
        vertices = []
        
        for i in range(latitudes + 1):
            theta = i * math.pi / latitudes
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            
            for j in range(longitudes + 1):
                phi = j * 2 * math.pi / longitudes
                sin_phi = math.sin(phi)
                cos_phi = math.cos(phi)
                
                x = cos_phi * sin_theta
                y = cos_theta
                z = sin_phi * sin_theta
                vertices.append([radius * x, radius * y, radius * z])
        
        # Tesselação dos triângulos da esfera
        for i in range(latitudes):
            for j in range(longitudes):
                v1 = i * (longitudes + 1) + j
                v2 = v1 + longitudes + 1
                gpu.GPU.draw_triangle(vertices[v1], vertices[v2], vertices[v1 + 1], colors)
                gpu.GPU.draw_triangle(vertices[v1 + 1], vertices[v2], vertices[v2 + 1], colors)


    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        slices = 20
        vertices = []
        for i in range(slices):
            theta = i * 2 * math.pi / slices
            x = bottomRadius * math.cos(theta)
            z = bottomRadius * math.sin(theta)
            vertices.append([x, 0, z])
        
        # Desenho da base
        for i in range(slices):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % slices]
            gpu.GPU.draw_triangle([0, 0, 0], v1, v2, colors)
        
        # Desenho das faces laterais
        top = [0, height, 0]
        for i in range(slices):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % slices]
            gpu.GPU.draw_triangle(v1, v2, top, colors)

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cylinder
        # A função cylinder é usada para desenhar cilindros na cena. O cilindro é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da base do cilindro e o argumento height especifica a altura do cilindro.
        # O cilindro é alinhado com o eixo Y local. O cilindro é fechado por padrão em ambas as extremidades.
        # Para desenha esse cilindro você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        slices = 20
        vertices_bottom = []
        vertices_top = []
        
        for i in range(slices):
            theta = i * 2 * math.pi / slices
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            vertices_bottom.append([x, 0, z])
            vertices_top.append([x, height, z])
        
        # Desenho das bases
        for i in range(slices):
            v1 = vertices_bottom[i]
            v2 = vertices_bottom[(i + 1) % slices]
            gpu.GPU.draw_triangle([0, 0, 0], v1, v2, colors)
            v1_top = vertices_top[i]
            v2_top = vertices_top[(i + 1) % slices]
            gpu.GPU.draw_triangle([0, height, 0], v1_top, v2_top, colors)
        
        # Desenho das faces laterais
        for i in range(slices):
            v1_bottom = vertices_bottom[i]
            v2_bottom = vertices_bottom[(i + 1) % slices]
            v1_top = vertices_top[i]
            v2_top = vertices_top[(i + 1) % slices]
            gpu.GPU.draw_triangle(v1_bottom, v2_bottom, v1_top, colors)
            gpu.GPU.draw_triangle(v2_bottom, v2_top, v1_top, colors)

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "NavigationInfo : headlight = {0}".format(headlight)
        )  # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color))  # imprime no terminal
        print(
            "DirectionalLight : intensity = {0}".format(intensity)
        )  # imprime no terminal
        print(
            "DirectionalLight : direction = {0}".format(direction)
        )  # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color))  # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity))  # imprime no terminal
        print("PointLight : location = {0}".format(location))  # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color))  # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "TimeSensor : cycleInterval = {0}".format(cycleInterval)
        )  # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = (
            time.time()
        )  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print(
            "SplinePositionInterpolator : key = {0}".format(key)
        )  # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]

        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key))  # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
