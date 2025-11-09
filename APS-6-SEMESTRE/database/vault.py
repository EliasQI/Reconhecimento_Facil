import sys
import os
import cv2
import face_recognition
import pickle
import numpy as np
from playsound import playsound
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QMessageBox
)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import QTimer, Qt


class CofreApp(QWidget):
    def __init__(self):
        super().__init__()

        # === CONFIG JANELA ===
        self.setWindowTitle("🔒 Sistema de Segurança do Cofre")
        self.resize(1000, 700)
        self.setStyleSheet("background-color: #101010; color: white;")

        # === ELEMENTOS VISUAIS ===
        self.video_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.status_label = QLabel("Inicializando reconhecimento facial...")
        self.status_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("background-color: #222; border-radius: 8px; padding: 12px;")

        self.img_status = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.img_status.setPixmap(QPixmap("aguardando.png").scaled(250, 250, Qt.AspectRatioMode.KeepAspectRatio))

        self.btn_sair = QPushButton("Encerrar Sistema")
        self.btn_sair.setFont(QFont("Arial", 14))
        self.btn_sair.setStyleSheet("""
            QPushButton {
                background-color: #444;
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """)
        self.btn_sair.clicked.connect(self.close)

        # === LAYOUT ===
        layout = QVBoxLayout()
        layout.addWidget(self.video_label, stretch=3)
        layout.addWidget(self.status_label)
        layout.addWidget(self.img_status, stretch=1)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.btn_sair)
        hbox.addStretch(1)
        layout.addLayout(hbox)
        self.setLayout(layout)

        # === INICIALIZA ===
        self.carregar_codificacoes()

        # tenta abrir câmera 0
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            # tenta outras câmeras (1–4)
            for i in range(1, 5):
                temp_cap = cv2.VideoCapture(i)
                if temp_cap.isOpened():
                    self.cap = temp_cap
                    break

        # se ainda assim falhar, mostra alerta e fecha
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Erro", "❌ Não foi possível acessar a câmera.\nVerifique as permissões do Windows.")
            sys.exit(1)

        self.timer = QTimer()
        self.timer.timeout.connect(self.atualizar_frame)
        self.timer.start(40)

        # evita tocar som repetido
        self.ultimo_status = None

    def carregar_codificacoes(self):
        print("Carregando banco de dados de rostos...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pkl_path = os.path.join(base_dir, "..", "codificacoes.pkl")

        if not os.path.exists(pkl_path):
            QMessageBox.critical(self, "Erro", f"Arquivo de codificações não encontrado:\n{pkl_path}")
            sys.exit(1)

        with open(pkl_path, "rb") as f:
            dados = pickle.load(f)

        self.codificacoes_conhecidas = dados["codificacoes"]
        self.metadados_conhecidos = dados["metadados"]
        print("Base carregada com sucesso!")

    def atualizar_frame(self):
        sucesso, frame = self.cap.read()

        # Frame vazio = câmera sem retorno
        if not sucesso or frame is None:
            print("⚠️ Frame vazio — câmera não retornou imagem.")
            return

        # Garante tipo numpy uint8
        frame = np.array(frame, dtype=np.uint8)

        try:
            # Converte sempre de BGR → RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        except Exception as e:
            print(f"Erro ao converter frame: {e}")
            return

        # Se ainda for inválido, ignora
        if frame_rgb is None or frame_rgb.dtype != np.uint8:
            print("⚠️ Frame incompatível — ignorando ciclo.")
            return

        # Reduz imagem (pra acelerar)
        frame_peq = cv2.resize(frame_rgb, (0, 0), fx=0.25, fy=0.25)
        frame_peq = np.ascontiguousarray(frame_peq, dtype=np.uint8)

        # 🧠 Debug: mostra formato e flags
        print("DEBUG:", frame_peq.shape, frame_peq.dtype, frame_peq.flags['C_CONTIGUOUS'])

        # 🔒 Protege chamada ao face_recognition
        try:
            locs = face_recognition.face_locations(frame_peq)
            encs = face_recognition.face_encodings(frame_peq, locs)
            self.erros_face = 0  # zera contador se der certo
        except Exception as e:
            if not hasattr(self, "erros_face"):
                self.erros_face = 0
            self.erros_face += 1
            print(f"Erro no face_recognition ({self.erros_face}): {e}")
            if self.erros_face > 5:
                print("❌ Muitos erros seguidos — pausando temporariamente.")
                self.timer.stop()
            return

        nome = "Desconhecido"
        acesso = "NEGADO"

        for enc in encs:
            matches = face_recognition.compare_faces(self.codificacoes_conhecidas, enc)
            dist = face_recognition.face_distance(self.codificacoes_conhecidas, enc)
            if True in matches:
                i = np.argmin(dist)
                meta = self.metadados_conhecidos[i]
                nome = meta["nome"]
                acesso = meta["nivel"].replace("_", " ").title()

        # Atualiza interface + toca som
        if nome == "Desconhecido":
            if self.ultimo_status != "negado":
                playsound("acesso_negado.mp3", block=False)
                self.ultimo_status = "negado"
            self.status_label.setText("🚫 Acesso Negado")
            self.status_label.setStyleSheet("background-color: #8b0000; color: white; padding: 10px; border-radius: 8px;")
            self.img_status.setPixmap(QPixmap("acesso_negado.png").scaled(250, 250, Qt.AspectRatioMode.KeepAspectRatio))
        else:
            if self.ultimo_status != "liberado":
                playsound("acesso_liberado.mp3", block=False)
                self.ultimo_status = "liberado"
            self.status_label.setText(f"✅ Acesso Liberado - {nome} ({acesso})")
            self.status_label.setStyleSheet("background-color: #006400; color: white; padding: 10px; border-radius: 8px;")
            self.img_status.setPixmap(QPixmap("acesso_liberado.png").scaled(250, 250, Qt.AspectRatioMode.KeepAspectRatio))

        # Exibe vídeo na interface
        qimg = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))



    def closeEvent(self, event):
        self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    janela = CofreApp()
    janela.show()
    sys.exit(app.exec())
