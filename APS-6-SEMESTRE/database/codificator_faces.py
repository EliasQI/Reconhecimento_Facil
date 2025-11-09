import face_recognition
import os
import pickle

print("Iniciando a codificação das faces...")

# Caminho para a base de dados
caminho_database = 'database'

# Listas para guardar as codificações e os metadados (nome e nível)
codificacoes_conhecidas = []
metadados_conhecidos = []

# Iterar sobre os níveis de acesso (nivel_1, nivel_2, etc.)
for nivel_acesso in os.listdir(caminho_database):
    caminho_nivel = os.path.join(caminho_database, nivel_acesso)
    
    if os.path.isdir(caminho_nivel):
        # Iterar sobre cada imagem de pessoa na pasta do nível
        for nome_arquivo in os.listdir(caminho_nivel):
            caminho_imagem = os.path.join(caminho_nivel, nome_arquivo)
            
            # Carregar a imagem
            imagem = face_recognition.load_image_file(caminho_imagem)
            
            # Extrair a codificação do rosto (pode haver mais de um rosto na imagem)
            # Pegamos a primeira codificação encontrada [0]
            codificacoes_rosto = face_recognition.face_encodings(imagem)
            
            if codificacoes_rosto:
                codificacao = codificacoes_rosto[0]
                
                # Adicionar a codificação à nossa lista
                codificacoes_conhecidas.append(codificacao)
                
                # Extrair o nome da pessoa do nome do arquivo
                nome_pessoa = os.path.splitext(nome_arquivo)[0].replace('_', ' ').title()
                
                # Guardar os metadados (nome e nível)
                metadados_conhecidos.append({
                    "nome": nome_pessoa,
                    "nivel": nivel_acesso
                })
                
                print(f"Rosto de {nome_pessoa} (Nível: {nivel_acesso}) codificado com sucesso.")

# Salvar as codificações e os metadados em um arquivo
dados_codificados = {"codificacoes": codificacoes_conhecidas, "metadados": metadados_conhecidos}

with open("codificacoes.pkl", "wb") as f:
    pickle.dump(dados_codificados, f)

print("\nCodificação finalizada! Os dados foram salvos em 'codificacoes.pkl'")