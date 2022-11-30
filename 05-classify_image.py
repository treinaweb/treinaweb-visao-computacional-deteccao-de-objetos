from imageai.Classification import ImageClassification
import os

path_execucao = os.getcwd()
predicao = ImageClassification()
predicao.setModelTypeAsResNet50()
predicao.setModelPath(os.path.join(path_execucao, "model/resnet50_imagenet_tf.2.0.h5"))
predicao.loadModel()

previsoes, percentual_probabilidade = predicao.classifyImage(
    os.path.join(path_execucao, "Img/image6.jpg"),
    result_count=5
)

for indice in range(len(previsoes)):
    print(previsoes[indice], " : ", percentual_probabilidade[indice])