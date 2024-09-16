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

        def sign(x):
            return (x > 0) - (x < 0)

        def insideTri(x, y, p1, p2, p3):
            d1 = sign((x - p2[0]) * (p1[1] - p2[1]) - (y - p2[1]) * (p1[0] - p2[0]))
            d2 = sign((x - p3[0]) * (p2[1] - p3[1]) - (y - p3[1]) * (p2[0] - p3[0]))
            d3 = sign((x - p1[0]) * (p3[1] - p1[1]) - (y - p1[1]) * (p3[0] - p1[0]))
            return (d1 == d2) and (d2 == d3)

        if len(vertices) < 6:
            raise ValueError("Invalid triangle vertex data.")

        color = np.array(colors.get("emissiveColor", [1.0, 1.0, 1.0])) * 255

        for i in range(0, len(vertices), 6):
            p1 = (vertices[i], vertices[i + 1])
            p2 = (vertices[i + 2], vertices[i + 3])
            p3 = (vertices[i + 4], vertices[i + 5])

            # Compute bounding box
            x_min, x_max = int(min(p1[0], p2[0], p3[0])), int(max(p1[0], p2[0], p3[0]))
            y_min, y_max = int(min(p1[1], p2[1], p3[1])), int(max(p1[1], p2[1], p3[1]))

            # Clamp bounding box to screen dimensions
            x_min = max(0, x_min)
            x_max = min(GL.width - 1, x_max)
            y_min = max(0, y_min)
            y_max = min(GL.height - 1, y_max)

            # Iterate over the bounding box
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    if insideTri(x + 0.5, y + 0.5, p1, p2, p3):
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)

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

        def insideTri(x, y, p1, p2, p3):
            """Check if the point (x, y) is inside the triangle defined by p1, p2, p3."""

            def edge_fn(v1, v2, p):
                return (p[0] - v2[0]) * (v1[1] - v2[1]) - (p[1] - v2[1]) * (
                    v1[0] - v2[0]
                )

            b1 = edge_fn(p1, p2, [x, y]) < 0.0
            b2 = edge_fn(p2, p3, [x, y]) < 0.0
            b3 = edge_fn(p3, p1, [x, y]) < 0.0

            return (b1 == b2) and (b2 == b3)

        def transform_points(points, screen_matrix):
            """Transform 3D points to 2D screen coordinates."""
            transformed_points = []

            for i in range(0, len(points), 3):
                p = np.array([points[i], points[i + 1], points[i + 2], 1.0])

                # Apply perspective, model, view transformations
                p = GL.perspective_matrix @ GL.transform_stack[-1] @ p
                p = p / p[-1]  # Perform perspective division

                # Transform to screen coordinates
                p = screen_matrix @ p

                transformed_points.append(p[0])
                transformed_points.append(p[1])

            return transformed_points

        # Set color
        color = np.array(colors.get("emissiveColor", [1.0, 1.0, 1.0])) * 255

        # Get the screen transformation matrix
        w, h = GL.width, GL.height
        screen_matrix = np.array(
            [[w / 2, 0, 0, w / 2], [0, -h / 2, 0, h / 2], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        # Transform points to 2D screen space
        vertices = transform_points(point, screen_matrix)

        # Iterate through each triangle (6 values per triangle)
        for i in range(0, len(vertices), 6):
            p1 = [vertices[i], vertices[i + 1]]
            p2 = [vertices[i + 2], vertices[i + 3]]
            p3 = [vertices[i + 4], vertices[i + 5]]

            # Compute bounding box for the triangle
            x_min, x_max = int(min(p1[0], p2[0], p3[0])), int(max(p1[0], p2[0], p3[0]))
            y_min, y_max = int(min(p1[1], p2[1], p3[1])), int(max(p1[1], p2[1], p3[1]))

            # Clamp bounding box to screen dimensions
            x_min = max(0, x_min)
            x_max = min(GL.width - 1, x_max)
            y_min = max(0, y_min)
            y_max = min(GL.height - 1, y_max)

            # Fill the triangle by iterating over the bounding box
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    if insideTri(x + 0.5, y + 0.5, p1, p2, p3):
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)

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

        # Scale matrix
        scale_m = np.array(
            [
                [scale[0], 0.0, 0.0, 0.0],
                [0.0, scale[1], 0.0, 0.0],
                [0.0, 0.0, scale[2], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # Translation matrix
        translation_m = np.array(
            [
                [1.0, 0.0, 0.0, translation[0]],
                [0.0, 1.0, 0.0, translation[1]],
                [0.0, 0.0, 1.0, translation[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # Rotation matrix
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

        # Combine transformations: Translate -> Rotate -> Scale
        object_to_world_m = translation_m @ rotation_m @ scale_m

        # Debugging print statement (can be removed in production)
        print("Model Matrix:\n", object_to_world_m)

        # Append the transformation to the stack
        GL.transform_stack.append(object_to_world_m)

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Saindo de Transform")

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
        # A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.

        color = np.array(colors.get("emissiveColor", [1.0, 1.0, 1.0])) * 255

        vertices = []
        for i in coordIndex:
            if i == -1:
                if len(vertices) >= 3:
                    GL.triangleSet(vertices, colors)
                vertices = []
            else:
                vertices.extend(coord[i * 3 : (i + 1) * 3])

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

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size))  # imprime no terminal pontos
        print("Box : colors = {0}".format(colors))  # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "Sphere : radius = {0}".format(radius)
        )  # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors))  # imprime no terminal as cores

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
