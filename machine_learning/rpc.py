from builtins import print
from itertools import chain
import cv2
import numpy as np
import glob
import tkinter as tk
import random
from load_model import get_prediction


# Képek beolvasása és feldolgozása adott könyvtárból
def process_data(path, mode, m=20):
    filenames = glob.glob(path)
    filenames.sort()
    images = [cv2.imread(img) for img in filenames]
    training_data = []
    i = 0
    for img in images:
        i = i + 1
        # Az első m mintát használjuk tanításra, a többit tesztelésre
        if mode == 'train' and i > m:
            break
        elif mode == 'test' and i <= m:
            continue

        features = extract_features(img)
        training_data.append(features)
    return training_data


def extract_features(img):
    # Előfeldolgozás: küszöbölés binarizálás - Otsu
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(imgray, (9, 9), 0)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Előfeldolgozás: morfológiai szűrések
    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.erode(thresh, struct)
    thresh = cv2.dilate(thresh, struct)
    thresh = cv2.dilate(thresh, struct)
    thresh = cv2.erode(thresh, struct)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_item = sorted_contours[0]
    # Csak a legnagyobb komponenset hagyjuk meg, mivel minden képen egyetlen objektumot feltételezünk.
    cnt = largest_item
    area = cv2.contourArea(cnt)

    # 1. jellemző: excentricitás (az objektumra illesztett ellipszisre vonatkozóan):
    # főtengely hossza / melléktengely hossza
    (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
    excentricity = MA / ma

    # 2. jellemző: tömörség = konvex burok területe / eredeti objektum területe
    ch = cv2.convexHull(cnt)
    solidity = cv2.contourArea(ch) / area

    # 3. jellemző: cirkularitás = terület / (kerület)2
    circularity = area / (cv2.arcLength(cnt, True) ** 2)

    # momentumok és hu-momentumok
    m = cv2.moments(cnt, True)
    hu = cv2.HuMoments(m)

    # print([excentricity, solidity, circularity])
    return [excentricity, solidity, circularity]


# Tanítás
def training():
    # Jellemző vektorok előállítása tanításhoz
    train_data = []

    # osztálycímkék feljegyzése
    labels = []

    for i in range(0, 3):
        print(f"{get_move(i)}...")
        # utolsó param: Mennyi mintával tanítjuk a modellt
        tdata = process_data(get_rps_path(i), 'train', 20)
        train_data.append(tdata)
        label = np.full(len(tdata), i)
        labels.append(label)

    labels = list(chain.from_iterable(labels))

    train_data = list(chain.from_iterable(train_data))

    train_data = np.float32(np.array(train_data))

    print("\nOsztályozási modell generálása:")
    # kNN osztályozó tanítása
    global knn
    knn = cv2.ml.KNearest_create()
    knn.train(train_data, cv2.ml.ROW_SAMPLE, np.array(labels))
    print("done")


def confusion_matrix(labels, responses):
    confusion = np.zeros((3, 3), np.int32)
    for i, j in zip(np.uint(labels), np.uint(responses)):
        confusion[i, j] += 1
    print('kNN tévesztési mátrix:')
    print(confusion, '\n')


# Tesztelés
def test_knn():
    test_data = []

    # osztálycímkék feljegyzése
    labels = []

    for i in range(0, 3):
        print(f"{get_move(i)}...")
        tdata = process_data(get_rps_path(i), 'test')
        test_data.append(tdata)
        label = np.full(len(tdata), i)
        labels.append(label)

    labels = list(chain.from_iterable(labels))

    test_data = list(chain.from_iterable(test_data))
    test_data = np.float32(np.array(test_data))

    features = np.float32(np.array(test_data))
    ret, responses, neighbours, dist = knn.findNearest(features, 9)

    correct_knn = sum(x == y for x, y in zip(labels, responses))
    print("\nkNN pontossága:", correct_knn[0] * 100.0 / len(labels), "%")

    confusion_matrix(labels, responses)


def clicked(player_move):
    # egy random kép a játékos döntése alapján
    filenames_p = glob.glob(get_rps_path(player_move))
    playerimgpath = random.choice(filenames_p)
    playerphoto['file'] = playerimgpath

    # GUI kiírás - játékos lépése
    lbl_actual["text"] = f"Játékos választása: {get_move(player_move)}"

    # random kép - a gép döntése
    machine_move = random.randint(0, 2)
    filenames_m = glob.glob(get_rps_path(machine_move))
    machineimgpath = random.choice(filenames_m)
    machinephoto['file'] = machineimgpath

    # GUI kiírás - gép lépése
    lbl_machine_move["text"] = f"Gép választása: {get_move(machine_move)}"

    # kő-papír-olló
    winner = rps_table[player_move][machine_move]
    if winner == player_move:
        lbl_winner_move["text"] = f"Nyertes: {get_move(winner)}, (Játékos)"
        player_score.set(player_score.get() + 1)
        lbl_player_score["foreground"] = "lime green"
        lbl_machine_score["foreground"] = "black"
    elif winner == machine_move:
        lbl_winner_move["text"] = f"Nyertes: {get_move(winner)}, (Gép)"
        machine_score.set(machine_score.get() + 1)
        lbl_machine_score["foreground"] = "lime green"
        lbl_player_score["foreground"] = "black"
    else:
        lbl_winner_move["text"] = "Döntetlen"

    knn_res_player = knn_prediction(playerimgpath, 9)
    knn_res_machine = knn_prediction(machineimgpath, 9)

    # if knn_res_player == player_move -> paint lbl_prediction green

    lbl_prediction["text"] = f"kNN jóslata: {get_move(knn_res_player)}"
    if knn_res_player == player_move:
        lbl_prediction["foreground"] = "green"
    else:
        lbl_prediction["foreground"] = "red"

    lbl_prediction_m["text"] = f"kNN jóslata: {get_move(knn_res_machine)}"
    if knn_res_machine == machine_move:
        lbl_prediction_m["foreground"] = "green"
    else:
        lbl_prediction_m["foreground"] = "red"

    neural_prediction_player = get_prediction(playerimgpath)

    lbl_neural_network_prediction["text"] = f"Neuronháló jóslata: {get_move(neural_prediction_player)}"
    if neural_prediction_player == player_move:
        lbl_neural_network_prediction["foreground"] = "green"
    else:
        lbl_neural_network_prediction["foreground"] = "red"

    neural_prediction_machine = get_prediction(machineimgpath)
    lbl_neural_network_prediction_m["text"] = f"Neuronháló jóslata: {get_move(neural_prediction_machine)}"
    if neural_prediction_machine == machine_move:
        lbl_neural_network_prediction_m["foreground"] = "green"
    else:
        lbl_neural_network_prediction_m["foreground"] = "red"

def knn_prediction(path_to_img, k):
    global knn
    img = cv2.imread(path_to_img)
    new_img_features = extract_features(img)
    features = np.float32(np.array([new_img_features]))
    ret, result, neighbours, dist = knn.findNearest(features, k)
    return int(result[0][0])


def get_rps_path(idx):
    paths = {
        0: "../datasets/1/rock/*.png",
        1: "../datasets/1/paper/*.png",
        2: "../datasets/1/scissors/*.png",
    }
    return paths.get(idx)


def get_move(idx):
    moves = {
        0: "Kő",
        1: "Papír",
        2: "Olló",
    }
    return moves.get(idx)


# Win-lose matrix
rps_table = [[-1, 1, 0], [1, -1, 2], [0, 2, -1]]


###########
# TANÍTÁS
###########

print("\nTanítás:")
training()

###########
# TESZTELÉS
###########

print("\nTesztelés...")
test_knn()


###########
# GUI
###########

# tkinter ablak
root = tk.Tk()
root.title("Kő-Papír-Olló")
root.resizable(width=False, height=False)

content = tk.Frame(root)

# Elért pontok
player_score = tk.IntVar(value=0)
machine_score = tk.IntVar(value=0)
lbl_player_score = tk.Label(content, font=('Verdana', 20), textvariable=player_score)
lbl_machine_score = tk.Label(content, font=('Verdana', 20), textvariable=machine_score)

# PhotoImage object - ikonok betöltése
icon_r = tk.PhotoImage(file="icons/rock.png")
icon_p = tk.PhotoImage(file="icons/paper.png")
icon_s = tk.PhotoImage(file="icons/scissors.png")

# Gombok - ikonok beállítása
btn_rock = tk.Button(content, text="ko", image=icon_r, command=lambda: clicked(0))
btn_paper = tk.Button(content, text="pap", image=icon_p, command=lambda: clicked(1))
btn_scissor = tk.Button(content, text="oll", image=icon_s, command=lambda: clicked(2))

# PhotoImage objects - a képek megjelenítéséhez
playerphoto = tk.PhotoImage()
machinephoto = tk.PhotoImage()
winnerphoto = tk.PhotoImage()

# Képek beállítása
lbl_player = tk.Label(content, image=playerphoto)
lbl_machine = tk.Label(content, image=machinephoto)
lbl_winner = tk.Label(content, image=winnerphoto)

# labels
lbl_prediction = tk.Label(content, font=('Verdana', 15), text="kNN jóslata: ")
lbl_prediction_m = tk.Label(content, font=('Verdana', 15), text="kNN jóslata: ")
lbl_neural_network_prediction = tk.Label(content, font=('Verdana', 15), text="Neuronháló jóslata: ")
lbl_neural_network_prediction_m = tk.Label(content, font=('Verdana', 15), text="Neuronháló jóslata: ")
lbl_actual = tk.Label(content, font=('Verdana', 15), text="Játékos választása: ")
lbl_machine_move = tk.Label(content, font=('Verdana', 15), text="Gép választása: ")
lbl_winner_move = tk.Label(content, font=('Verdana', 13), text="Nyertes: ")

# Grid
content.grid(column=0, row=0)
lbl_player_score.grid(column=1, row=0)
lbl_machine_score.grid(column=2, row=0)
btn_rock.grid(column=0, row=1)
btn_paper.grid(column=0, row=2)
btn_scissor.grid(column=0, row=3)
lbl_player.grid(column=1, row=1, rowspan=3, pady=5, padx=5)
lbl_machine.grid(column=2, row=1, rowspan=3, pady=5, padx=5)
lbl_winner.grid(column=1, row=4, columnspan=2, pady=5, padx=5)
lbl_actual.grid(column=1, row=5, sticky="w")
lbl_prediction.grid(column=1, row=6, sticky="w")
lbl_prediction_m.grid(column=2, row=6, sticky="w")
lbl_neural_network_prediction.grid(column=1, row=7, sticky="w")
lbl_neural_network_prediction_m.grid(column=2, row=7, sticky="w")
lbl_machine_move.grid(column=2, row=5, sticky="w")
lbl_winner_move.grid(column=0, row=5, sticky="w")

content.columnconfigure(0, minsize=300)
content.columnconfigure([1, 2], minsize=300)

root.mainloop()
