# realtime_recognize.py
import cv2
import face_recognition
import numpy as np
import sqlite3
import pickle
import time

THRESHOLD = 0.55  # sensibilidade

def load_db():
    conn = sqlite3.connect('faces.db')
    cur = conn.cursor()
    cur.execute("SELECT id, name, permissions FROM users")
    users = {row[0]: {"name": row[1], "permissions": row[2].split(",")} for row in cur.fetchall()}
    cur.execute("SELECT user_id, vector FROM embeddings")
    embeddings = {}
    for uid, blob in cur.fetchall():
        vec = pickle.loads(blob)
        if uid not in embeddings:
            embeddings[uid] = []
        embeddings[uid].append(vec)
    conn.close()
    return users, embeddings

def find_best_match(embedding, db_embeddings):
    best_id = None
    best_dist = float("inf")
    for uid, vecs in db_embeddings.items():
        for v in vecs:
            dist = np.linalg.norm(v - embedding)
            if dist < best_dist:
                best_dist = dist
                best_id = uid
    return best_id, best_dist

def main():
    users, embeddings = load_db()
    if not users:
        print("❌ Nenhum usuário cadastrado.")
        return

    cam = cv2.VideoCapture(0)
    print("Iniciando reconhecimento facial... Pressione [ESC] para sair.")
    last_recognized = None
    last_time = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encs = face_recognition.face_encodings(rgb, boxes)

        for (top, right, bottom, left), enc in zip(boxes, encs):
            uid, dist = find_best_match(enc, embeddings)
            if uid and dist < THRESHOLD:
                name = users[uid]["name"]
                perms = users[uid]["permissions"]
                label = f"{name} ({dist:.2f})"
                color = (0, 255, 0)
            else:
                name, perms, label = "Desconhecido", [], "Desconhecido"
                color = (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Perms: {', '.join(perms)}", (left, bottom + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

            # lógica de permissões
            if name != "Desconhecido":
                now = time.time()
                if name != last_recognized or now - last_time > 5:
                    print(f"[{time.ctime()}] {name} reconhecido - Permissões: {perms}")
                    if "admin" in perms:
                        print("🔓 Acesso total liberado")
                    elif "laboratorio" in perms:
                        print("🧪 Acesso apenas ao laboratório")
                    else:
                        print("🚫 Permissão restrita")
                    last_recognized, last_time = name, now

        cv2.imshow("Reconhecimento Facial", frame)
        if cv2.waitKey(1) % 256 == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()