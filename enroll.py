import cv2
import face_recognition
import numpy as np
import sqlite3
import pickle
import datetime

def embed_images(images):
    embeddings = []
    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encs = face_recognition.face_encodings(rgb, boxes)
        if len(encs) > 0:
            embeddings.append(encs[0])
    return embeddings

def capture_images(num_images=5):
    cam = cv2.VideoCapture(0)
    imgs = []
    count = 0
    print("Pressione [ESPAÇO] para capturar imagem, [ESC] para sair.")
    while count < num_images:
        ret, frame = cam.read()
        if not ret:
            break
        cv2.putText(frame, f"{count}/{num_images}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Cadastro", frame)
        k = cv2.waitKey(1)
        if k % 256 == 27:
            break
        elif k % 256 == 32:
            imgs.append(frame.copy())
            count += 1
            print(f"Imagem {count} capturada.")
    cam.release()
    cv2.destroyAllWindows()
    return imgs

def save_to_db(user_id, name, permissions, embeddings):
    conn = sqlite3.connect('faces.db')
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO users (id, name, permissions, created_at) VALUES (?, ?, ?, ?)",
                (user_id, name, ','.join(permissions), str(datetime.datetime.now())))
    for emb in embeddings:
        cur.execute("INSERT INTO embeddings (user_id, vector) VALUES (?, ?)", (user_id, pickle.dumps(emb)))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    user_id = input("ID do usuário: ").strip()
    name = input("Nome: ").strip()
    perms = input("Permissões (separadas por vírgula): ").strip().split(",")
    imgs = capture_images()
    embeddings = embed_images(imgs)
    if embeddings:
        save_to_db(user_id, name, perms, embeddings)
        print(f"✅ {name} cadastrado com sucesso!")
    else:
        print("❌ Nenhum rosto detectado. Tente novamente.")