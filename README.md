Eindwerk NLP model maken met Tensorflow door Niels Delanghe, Simon Luyten, Gleb Shalabotov en Sander Vanderlinden.

In de map website staan alle files voor het visualiseren van onze resultaten. Er ontbreken wel 3 model files, deze zijn glove_f300.txt, ft150.bin en ft300.bin. Deze files ontbreken omdat ze te groot zijn om naar git te sturen, ook wanneer ze gezipet zijn en via git lfs worden doorgestuurd.

Scripts om de dataset te bewerken:
https://github.com/SanderVanderlinden/Tensorflow/tree/master/nlcow%20formatting

gapfill model classifier:
https://github.com/SanderVanderlinden/Tensorflow/tree/master/model_met_woord_output

gapfill model met vector output:
https://github.com/SanderVanderlinden/Tensorflow/tree/master/model_met_vector_output/gapfill

Lorem Ipsum model train file (bepaalde waarden aan te passen voor de locatie van het corpus)
https://github.com/SanderVanderlinden/Tensorflow/blob/master/Simon/nice.py

In deze Simon folder zijn ook de files die ik gebruikt heb om telken als ik een model getrained had dat te testen ( main.py)
en alsook het .pbs script voor mijn nice.p te runnen op de Supercomputer :)


