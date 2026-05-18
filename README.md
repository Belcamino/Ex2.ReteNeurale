# Dal libro Arithmós 
*Tempo · Relatività · Quanti · Intelligenze
Dai pitagorici alla coscienza artificiale*

- [MondadoriStore](https://www.mondadoristore.it/arithmos-tempo-relativita-quanti-intelligenze-dai-pitagorici-alla-coscienza-artificiale-libro-luigi-belcamino/p/9791224076780 "MondadoriStore")
- [Amazon](https://amzn.eu/d/0gmjyQnW "Amazon")
- [Youcanprint Store](https://store.youcanprint.it/arithmos-tempo-relativita-quanti-intelligenze-dai-pitagorici-alla-coscienza-artificiale/b/d4089bb3-c7ba-5343-921f-9c353ddee1ff "Youcanprint Store")
 

## Esercitazione Ex2.reteneurale
Il codice *Python* creerà e addestrerà una semplice rete neurale per gestire un problema di regressione lineare.
Dato un certo dataset di punti su un piano cartesiano, la rete neurale addestrata sarà in grado di riconoscere l’andamento. 

Useremo ``TensorFlow/Keras`` per definire e addestrare il modello. 

Dopo l’importazione delle librerie necessarie, il programma stampa a video la versione della libreria ``tensorflow`` utilizzata e crea un dataset semplice con una relazione lineare tra input (X) e output (y), aggiungendo un po' di rumore per simulare dati reali. 

Poi il programma definisce una rete neurale sequenziale molto semplice con un solo strato nascosto (*dense Layer*) e uno strato di output. Dato che è un problema di regressione, lo strato di output non avrà una funzione di attivazione. 

Prima di addestrare il modello, se ne esegue la compilazione, ovvero una serie di operazioni di ottimizzazione (l'algoritmo usato per aggiornare i pesi della rete è *adam*), si definisce una funzione di *loss* per misurare quanto bene il modello sta performando e si definisce una metrica per monitorare le prestazioni durante l'addestramento. 

Poi si addestra il modello utilizzando il metodo ``fit()``. 

Forniamo i dati di input (X), i dati di output target (y), il numero di epoche (quante volte il modello vedrà l'intero *dataset*) e la dimensione del batch. 

Dopo l'addestramento, possiamo usare il modello per fare predizioni su nuovi dati e valutarne le prestazioni.

