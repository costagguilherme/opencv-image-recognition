import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import zipfile

# print(cv2.__version__)


def mostrar(img):
    figura = plt.gcf()
    figura.set_size_inches(16, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


labels_path = os.path.sep.join(["./files", "coco.names"])
weights_path = os.path.sep.join(["./files", "yolov4.weights"])
config_path = os.path.sep.join(["./files", "yolov4.cfg"])

LABELS = open(labels_path).read().strip().split("\n")
# print(LABELS)

net = cv2.dnn.readNet(config_path, weights_path)  ## carregando rede neural

tamanho = len(LABELS)
# Definindo a cor para o bound box de cada uma das classes
COLORS = np.random.randint(0, 255, size=(tamanho, 3), dtype="uint8")

ln = net.getLayerNames()


nomeCamadasDeSaida = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

imgPath = "./imgs/cat.jpeg"
image = cv2.imread(imgPath)

imgCopy = image.copy()
(H, W) = image.shape[:2]

# PROCESSAMENTO

tempoInicio = time.time()

# A imagem deve ser convertida para o formato blob antes de ser enviada para a rede neural
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

net.setInput(blob)  # Enviando img para a rede
layer_outputs = net.forward(nomeCamadasDeSaida)  # resultados
tempoTermino = time.time()


threshold = 0.5

caixas = []  # desenha bound box
confiancas = []
idClasses = []


for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        classeIndex = np.argmax(scores)  # indice da classe com maior probabilidade
        confianca = scores[classeIndex]
        if confianca > threshold:
            caixa = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = caixa.astype("int")

            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            caixas.append([x, y, int(width), int(height)])
            confiancas.append(float(confianca))
            idClasses.append(classeIndex)


threshold_NMS = 0.1
objs = cv2.dnn.NMSBoxes(caixas, confiancas, threshold, threshold_NMS)
print(objs)
for i in range(len(objs)):
    (x, y) = (caixas[i][0], caixas[i][1])
    (w, h) = (caixas[i][2], caixas[i][3])

    objeto = imgCopy[y : y + h, x : x + w]
    # cv2_imshow(objeto)

    cor = [int(c) for c in COLORS[idClasses[i]]]

    cv2.rectangle(image, (x, y), (x + w, y + h), cor, 2)
    texto = f"{LABELS[idClasses[i]]} {round(confiancas[objs[i]], 3)}"
    cv2.putText(image, texto, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
    cv2.imwrite("resultado.jpg", image)

mostrar(image)
